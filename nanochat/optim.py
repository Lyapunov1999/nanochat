"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
Two versions are provided (MuonAdamW, DistMuonAdamW), for single GPU and distributed.

Addapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

import math
import torch
import torch.optim as optim
import torch.distributed as dist
from torch import Tensor
from nanochat.common import COMPUTE_DTYPE
from nanochat.linesearch import strong_wolfe, armijo, linesearch

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,              # (32768, 768) - parameter tensor
    grad: Tensor,           # (32768, 768) - gradient, same shape as p
    exp_avg: Tensor,        # (32768, 768) - first moment, same shape as p
    exp_avg_sq: Tensor,     # (32768, 768) - second moment, same shape as p
    step_t: Tensor,         # () - 0-D CPU tensor, step count
    lr_t: Tensor,           # () - 0-D CPU tensor, learning rate
    beta1_t: Tensor,        # () - 0-D CPU tensor, beta1
    beta2_t: Tensor,        # () - 0-D CPU tensor, beta2
    eps_t: Tensor,          # () - 0-D CPU tensor, epsilon
    wd_t: Tensor,           # () - 0-D CPU tensor, weight decay
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    """
    # Weight decay (decoupled, applied before the update)
    p.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # Bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
https://arxiv.org/pdf/2510.05491

Some of the changes in nanochat implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
"""

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    """

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express
    # Cast to bf16 for speed when available; skip cast otherwise (fp16 is unstable here due to limited exponent range)
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------
# Single GPU version of the MuonAdamW optimizer.
# Used mostly for reference, debugging and testing.

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU version.

    AdamW - Fused AdamW optimizer step.

    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        # AdamW tensors
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon tensors
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        """
        AdamW update for each param in the group individually.
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1

            # Fill 0-D tensors with current values
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])

            # Fused update: weight_decay -> momentum -> bias_correction -> param_update
            adamw_step_fused(
                p, grad, exp_avg, exp_avg_sq,
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        """
        Muon update for all params in the group (stacked for efficiency).
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        params: list[Tensor] = group['params']
        if not params:
            return

        # Get or create group-level buffers (stored in first param's state for convenience)
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        # Momentum for every individual parameter
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        # Second momentum buffer is factored, either per-row or per-column
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Stack grads and params (NOTE: this assumes all params have the same shape)
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        # Fill all the 0-D tensors with current values
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        # Single fused kernel: momentum -> polar_express -> variance_reduction -> update
        muon_step_fused(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )

        # Copy back to original params
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

# -----------------------------------------------------------------------------
# Distributed version of the MuonAdamW optimizer.
# Used for training on multiple GPUs.

class DistMuonAdamW(torch.optim.Optimizer):
    """
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.

    See MuonAdamW for the algorithmic details of each optimizer. This class adds
    distributed communication to enable multi-GPU training without PyTorch DDP.

    Design Goals:
    - Overlap communication with computation (async ops)
    - Minimize memory by sharding optimizer states across ranks (ZeRO-2 style)
    - Batch small tensors into single comm ops where possible

    Communication Pattern (3-phase async):
    We use a 3-phase structure to maximize overlap between communication and compute:

        Phase 1: Launch all async reduce ops
            - Kick off all reduce_scatter/all_reduce operations
            - Don't wait - let them run in background while we continue

        Phase 2: Wait for reduces, compute updates, launch gathers
            - For each group: wait for its reduce, compute the update, launch gather
            - By processing groups in order, earlier gathers run while later computes happen

        Phase 3: Wait for gathers, copy back
            - Wait for all gathers to complete
            - Copy updated params back to original tensors (Muon only)

    AdamW Communication (ZeRO-2 style):
    - Small params (<1024 elements): all_reduce gradients, update full param on each rank.
      Optimizer state is replicated but these params are tiny (scalars, biases).
    - Large params: reduce_scatter gradients so each rank gets 1/N of the grad, update
      only that slice, then all_gather the updated slices. Optimizer state (exp_avg,
      exp_avg_sq) is sharded - each rank only stores state for its slice.
      Requires param.shape[0] divisible by world_size.

    Muon Communication (stacked + chunked):
    - All params in a Muon group must have the same shape (caller's responsibility).
    - Stack all K params into a single (K, *shape) tensor for efficient comm.
    - Divide K params across N ranks: each rank "owns" ceil(K/N) params.
    - reduce_scatter the stacked grads so each rank gets its chunk.
    - Each rank computes Muon update only for params it owns.
    - all_gather the updated params back to all ranks.
    - Optimizer state (momentum_buffer, second_momentum_buffer) is sharded by chunk.
    - Padding: if K doesn't divide evenly, we zero-pad to (ceil(K/N) * N) for comm,
      then ignore the padding when copying back.

    Buffer Reuse:
    - For Muon, we allocate stacked_grads for reduce_scatter input, then reuse the
      same buffer as the output for all_gather (stacked_params). This saves memory
      since we don't need both buffers simultaneously.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """Launch async reduce ops for AdamW group. Returns info dict with per-param infos."""
        param_infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                # Small params: all_reduce (no scatter/gather needed)
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                # Large params: reduce_scatter
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        """Launch async reduce op for Muon group. Returns info dict."""
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # Stack grads and zero-pad to padded_num_params
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        # Reduce_scatter to get this rank's chunk
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        """Wait for reduce, compute AdamW updates, launch gathers for large params."""
        param_infos = info['param_infos']
        for p in group['params']:
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]

            # For small params, operate on full param; for large, operate on slice
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1

            # Fill 0-D tensors and run fused kernel
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

            # Large params need all_gather
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """Wait for reduce, compute Muon updates, launch gather."""
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # How many params does this rank own?
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        # Get or create group-level state
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Build output buffer for all_gather
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            # Fill 0-D tensors and run fused kernel
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        # Reuse stacked_grads buffer for all_gather output
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        """Wait for all gathers and copy Muon params back."""
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                # Muon: copy from stacked buffer back to individual params
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Phase 1: launch all async reduce ops
        reduce_infos: list[dict] = []
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'muon':
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 2: wait for reduces, compute updates, launch gathers
        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon':
                self._compute_muon(group, info, gather_list, rank)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 3: wait for gathers, copy back
        self._finish_gathers(gather_list)


class OptimizerWithLineSearch():
    def __init__(self, 
     ls_opt,
     fixed_opt,
     c1: float = 1e-4, # Armijo parameter
     c2: float = 0.9, # curvature parameter
     defaultLR_max_mult: float = 1e5, # bounds multiple of defaultLR the line-search LR can be
     defaultLR_min_mult: float = 1e-1,
     max_iter: int = 1, # number of iterations per step; note PyTorch's l-BFGS defaults to 20
     per_parameter: bool = False, # process parameters separately (True) or as a group (False)
     pos_only: bool = False, # defaults to using both positive and negative LRs
     monotone_strength: float= 1.0, # Zhang and Hager's \eta_k: 0=monotone, 1=A_k (average)
     ls_type: str ="strong_wolfe",
     extraLs_opt: optim=None # extra optimizer for stem + head style partial LS
    ):
        #if not isinstance(ls_opt, (optim.SGD, optim.AdamW)):
        #    raise TypeError("Only SGD and AdamW are supported for now.")   
        self.ls_opt = ls_opt
        self.fixed_opt = fixed_opt
        self.extraLs_opt = extraLs_opt
        self.per_parameter = per_parameter
        self.c1 = c1
        self.c2 = c2
        if defaultLR_max_mult is None:
            defaultLR_max_mult = 1e5
        if defaultLR_min_mult is None:
            defaultLR_min_mult = 1e-1
        self.defaultLR_max_mult = defaultLR_max_mult
        self.defaultLR_min_mult = defaultLR_min_mult
        self.max_iter = max_iter
        self.pos_only = pos_only
        self.monotone_strength = monotone_strength
        self.ls_type = ls_type
        self.Ck = float("inf")
        self.Qk = 0
        self.step_count = 0

    @property
    def param_groups(self):
        return self.ls_opt.param_groups

    def state_dict(self):
        return {
            "ls_opt": self.ls_opt.state_dict(),
            "fixed_opt": self.fixed_opt.state_dict(),
            "extraLs_opt": None if self.extraLs_opt is None else self.extraLs_opt.state_dict(),
            "wrapper_state": {
                "per_parameter": self.per_parameter,
                "c1": self.c1,
                "c2": self.c2,
                "defaultLR_max_mult": self.defaultLR_max_mult,
                "defaultLR_min_mult": self.defaultLR_min_mult,
                "max_iter": self.max_iter,
                "pos_only": self.pos_only,
                "monotone_strength": self.monotone_strength,
                "ls_type": self.ls_type,
                "Ck": self.Ck,
                "Qk": self.Qk,
                "step_count": self.step_count,
            },
        }

    def load_state_dict(self, state_dict):
        self.ls_opt.load_state_dict(state_dict["ls_opt"])
        self.fixed_opt.load_state_dict(state_dict["fixed_opt"])
        extra_state = state_dict.get("extraLs_opt")
        if self.extraLs_opt is not None and extra_state is not None:
            self.extraLs_opt.load_state_dict(extra_state)

        wrapper_state = state_dict.get("wrapper_state", {})
        self.Ck = wrapper_state.get("Ck", self.Ck)
        self.Qk = wrapper_state.get("Qk", self.Qk)
        if "step_count" in wrapper_state:
            self.step_count = int(wrapper_state["step_count"])
        elif self.ls_opt.param_groups and self.ls_opt.param_groups[0]["params"]:
            first_param = self.ls_opt.param_groups[0]["params"][0]
            legacy_step = self.ls_opt.state.get(first_param, {}).get("step")
            if legacy_step is not None:
                legacy_step = legacy_step.item() if torch.is_tensor(legacy_step) else legacy_step
                self.step_count = int(legacy_step) + 1

    @torch.no_grad()
    def get_direction(self, params, group, closure=None):
        if len(self.ls_opt.param_groups) != 1:
            raise NotImplementedError("Wolfe line search currently supports one line-search parameter group")
        if group is not self.ls_opt.param_groups[0]:
            raise NotImplementedError("Extra line-search optimizer groups are not supported")
        if len(params) != len(group["params"]) or any(p is not gp for p, gp in zip(params, group["params"])):
            raise ValueError("Directions must be generated for the complete line-search parameter group")

        params_before = self._clone_param(params)
        # The initial closure has already populated gradients. Stepping without a
        # closure advances momentum once and avoids an extra forward/backward pass.
        self.ls_opt.step()
        directions = [((p - p_before)/group['lr']).view(-1) for p_before,p in zip(params_before, params)]
        self._set_param(params, params_before)
        return torch.cat(directions, 0) # vectorize

    def __set_state_defaults(self, state):
        state.setdefault('n_iter', 0)
        state.setdefault('done', False)
        state.setdefault('prev_grad', None)
        state.setdefault('prev_d', None)
        state.setdefault('prev_f', float('inf'))
        state.setdefault('n_pos_lr', 0)
        state.setdefault('n_neg_lr', 0)
        state.setdefault('prev_gTd', 0)
        state.setdefault('numel_cache', None)
        state.setdefault('prev_lr', 0)
    
    def __update_state(self, state, loss, lr, d, grad, gTd):
        state['prev_f'] = float(loss)
        state['prev_lr'] = lr
        #print(f"__update_state: lr={lr}")
        state['prev_gTd'] = gTd
        if lr < 0:
            state['n_neg_lr'] += 1
        else:
            state['n_pos_lr'] += 1
        if state['prev_d'] is None:
            state['prev_d'] = d.clone(memory_format=torch.contiguous_format)
        else:
            state['prev_d'].copy_(d)
        if state['prev_grad'] is None:
            state['prev_grad'] = grad.clone(memory_format=torch.contiguous_format)
        else:
            state['prev_grad'].copy_(grad)
        state['n_iter'] += 1

    def __set_group_defaults(self, group):
        #print("__set_group_defaults")
        group['n_iter'] = 0
        group['done'] = False
        group['prev_grad'] = None
        group['prev_d'] = None
        group['prev_f'] = float('inf')
        group['n_pos_lr'] = 0
        group['n_neg_lr'] = 0
        group['prev_gTd'] = 0
        group['numel_cache'] = None
        group['prev_lr'] = 0
    
    def __update_group(self, group, loss, lr, d, grad, gTd):
        group['prev_f'] = float(loss)
        group['prev_lr'] = lr
        #print(f"update_group: lr = {lr}, prev_lr = {group['prev_lr']}")
        group['prev_gTd'] = gTd
        if lr < 0:
            group['n_neg_lr'] += 1
        else:
            group['n_pos_lr'] += 1
        if group['prev_d'] is None:
            group['prev_d'] = d.clone(memory_format=torch.contiguous_format)
        else:
            group['prev_d'].copy_(d)
        if group['prev_grad'] is None:
            group['prev_grad'] = grad.clone(memory_format=torch.contiguous_format)
        else:
            group['prev_grad'].copy_(grad)
        group['n_iter'] += 1    

    def __update_nonmonotone_reference(self, loss):
        loss = float(loss)
        if self.Ck == float('inf'):
            self.Ck = loss
            self.Qk = 1
            return
        self.Ck = self.monotone_strength * self.Qk * self.Ck + loss
        self.Qk = self.monotone_strength * self.Qk + 1
        self.Ck /= self.Qk

    # taken from PyTorch's l-BFGS implementation
    def _numel(self, params, numel_cache):
        if numel_cache is None:
            numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in params
            )
        return numel_cache

    # taken from PyTorch's l-BFGS implementation
    def _gather_flat_grad(self, params):
        views = []
        for p_idx, p in enumerate(params):
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        return torch.cat(views, 0)

    # taken from PyTorch's l-BFGS implementation
    def _add_grad(self, params, numel_cache, step_size, update):
        offset = 0
        for p in params:
            if torch.is_complex(p):
                p = torch.view_as_real(p)
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel(params, numel_cache)

    # taken from PyTorch's l-BFGS implementation
    def _clone_param(self, params):
        return [p.clone(memory_format=torch.contiguous_format) for p in params]
    
    # taken from PyTorch's l-BFGS implementation
    def _set_param(self, params, params_data):
        for p, pdata in zip(params, params_data):
            p.copy_(pdata)
    
    # print debug information for per-layer parameters
    def __print_debug_for_total(self, pList):
        n_pos_lr, n_neg_lr = 0, 0
        min_gTd, max_gTd = float('inf'), -float('inf')
        min_lr, max_lr = float('inf'), -float('inf')
        for p in pList:
            state = self.ls_opt.state[p]
            n_pos_lr += state['n_pos_lr']
            n_neg_lr += state['n_neg_lr']
            gTd, lr = state['prev_gTd'], state['prev_lr']
            if gTd < min_gTd:
                min_gTd = gTd
            if gTd > max_gTd:
                max_gTd = gTd
            if lr < min_lr:
                min_lr = lr
            if lr > max_lr:
                max_lr = lr
        print(f" n pos lr={n_pos_lr}, n neg lr={n_neg_lr}, gTd=[{min_gTd:>0.4f},"
        f" {max_gTd:>0.4f}], lr=[{min_lr:>0.4f}, {max_lr:>0.4f}]")

    # this version is for per-layer parameters
    #  returns the step size and directional derivative given search direction d
    #  and parameter p using either l-BFGS's strong wolfe line search (default)
    #  or a simpler implementation of strong wolfe (without cubic interpolation to
    #  find next trial step size)
    def __stepsize(self, p, numel_cache, closure, d, grad, loss, 
     c1, c2, defaultLR, update_reference=True):#, lsType="strong_wolfe"):
        gTd = torch.dot(d,grad)

        def objGradFunc(p_fixed, t, d):
            self._add_grad([p], numel_cache, t, d)
            f_new = closure()
            g_new = self._gather_flat_grad([p])
            self._set_param([p], [p_fixed]) 
            return float(f_new), g_new
        
        def objFunc(p_fixed, t, d):
            self._add_grad([p], numel_cache, t, d)
            f_new = closure()
            self._set_param([p], [p_fixed])
            return float(f_new)
        
        if self.ls_type == "strong_wolfe":
            f = loss.item()
            reference_f = f if self.Ck == float('inf') else self.Ck
            if gTd > 0 and not self.pos_only:
                f_new, g_new, lr, ls_func_evals = strong_wolfe(objGradFunc, 
                    p.clone(memory_format=torch.contiguous_format), 
                    defaultLR, d.neg(), f, grad, -gTd, c1=c1, c2=c2, non_monotone_f=reference_f,
                    min_t=self.defaultLR_min_mult*defaultLR, max_t=self.defaultLR_max_mult*defaultLR)
                if math.isnan(lr):
                    lr = 0
                    f_new = f
                lr *= -1
                #print(f" gTd={gTd:>0.4f}, lr={lr:>0.4f}, f={f:0.4f}, f_new={f_new:>0.4f}, ls_func_evals={ls_func_evals}")
            else:
                f_new, g_new, lr, ls_func_evals = strong_wolfe(objGradFunc, 
                    p.clone(memory_format=torch.contiguous_format), 
                    defaultLR, d, f, grad, gTd, c1=c1, c2=c2, non_monotone_f=reference_f,
                    min_t=self.defaultLR_min_mult*defaultLR, max_t=self.defaultLR_max_mult*defaultLR) 
                if math.isnan(lr):
                    lr =0
                    f_new = f
                #print(f" gTd={gTd:>0.4f}, lr={lr:>0.4f}, f={f:0.4f}, f_new={f_new:>0.4f}, ls_func_evals={ls_func_evals}")
            if update_reference:
                self.__update_nonmonotone_reference(f_new)
        elif self.ls_type == "simple_strong_wolfe":
            if gTd > 0 and not self.pos_only:
                lr = -1.0* linesearch(objGradFunc, d.neg(), grad, loss, -gTd, 
                    p.clone(memory_format=torch.contiguous_format), c1=c1, c2=c2)
            else:
                lr = linesearch(objGradFunc, d, grad, loss, gTd,
                    p.clone(memory_format=torch.contiguous_format), c1=c1, c2=c2)
        elif self.ls_type == "armijo":
            f = loss.item()
            if self.Ck==float('inf'):
                self.Ck=f
            if gTd > 0 and not self.pos_only:
                lr, f_new = -1.0* armijo(objFunc, d.neg(), grad, max(self.Ck, f), defaultLR,
                    p.clone(memory_format=torch.contiguous_format), -gTd, c1=c1)
            else:
                lr, f_new = armijo(objFunc, d, grad, max(self.Ck, f), defaultLR,
                    p.clone(memory_format=torch.contiguous_format), gTd, c1=c1)
            self.Ck = self.monotone_strength* self.Qk* self.Ck + f_new
            self.Qk = self.monotone_strength* self.Qk + 1
            self.Ck/= self.Qk
        else:
            lr = defaultLR
            print(f"Unrecognized lsType {self.ls_type}. Using default LR={lr}.")

        return lr, gTd
    
    # this version is for no per-layer parameters 
    #  returns the step size and directional derivative given search direction d
    #  and list of parameters pList using either l-BFGS's strong wolfe line search (default)
    #  or a simpler implementation of strong wolfe (without cubic interpolation to
    #  find next trial step size)
    def __stepsize_no_per_layer(self, pList, numel_cache, closure, d, grad, loss, 
     c1, c2, defaultLR, update_reference=True):#, lsType="strong_wolfe"):
        gTd = torch.dot(d, grad)

        def objGradFunc(x, t, d):
            self._add_grad(pList, numel_cache, t, d)
            f_new = closure()
            g_new = self._gather_flat_grad(pList)
            self._set_param(pList, x)
            return float(f_new), g_new
    
        def objFunc(x, t, d):
            self._add_grad(pList, numel_cache, t, d)
            f_new = closure()
            self._set_param(pList, x)
            return float(f_new)

        pfixed_list = self._clone_param(pList)
        if self.ls_type == "strong_wolfe":
            f = loss.item()
            reference_f = f if self.Ck == float('inf') else self.Ck
            #print(f"__stepsize_no_per_layer: gTd = {gTd}, f_prev = {f}, Q={self.Qk}, C={self.Ck}")
            if gTd > 0 and not self.pos_only:
                f_new, g_new, lr, ls_func_evals = strong_wolfe(objGradFunc, pfixed_list, 
                    defaultLR, d.neg(), f, grad, -gTd, c1=c1, c2=c2, non_monotone_f=reference_f,
                    min_t=self.defaultLR_min_mult*defaultLR, max_t=self.defaultLR_max_mult*defaultLR)
                #lr_check = lr
                f_new_check = f_new
                if math.isnan(lr):
                    lr = 0
                    f_new = f
                lr *= -1
                #print(f" gTd={gTd:>0.4f}, lr={lr_check:>0.4f}-->{lr:>0.4f}, f={f:>0.4f}, f_new={f_new_check:>0.4}-->{f_new:>0.4f}, ls_func_evals={ls_func_evals}")
            else:
                f_new, g_new, lr, ls_func_evals = strong_wolfe(objGradFunc, pfixed_list, 
                    defaultLR, d, f, grad, gTd, c1=c1, c2=c2, non_monotone_f=reference_f,
                    min_t=self.defaultLR_min_mult*defaultLR, max_t=self.defaultLR_max_mult*defaultLR)
                #lr_check = lr
                f_new_check = f_new
                if math.isnan(lr):
                    lr = 0
                    f_new = f
                #print(f" gTd={gTd:>0.4f}, lr={lr_check:>0.4f}-->{lr:>0.4f}, f={f:0.4f}, f_new={f_new_check:>0.4}-->{f_new:>0.4f}, ls_func_evals={ls_func_evals}")
            if update_reference:
                self.__update_nonmonotone_reference(f_new)

            #print(f"new C = {self.Ck}, new Q = {self.Qk}")
        elif self.ls_type == "simple_strong_wolfe":
            if gTd > 0 and not self.pos_only:
                lr = -1.0* linesearch(objGradFunc, d.neg(), grad, loss, -gTd, pfixed_list, 
                    c1=c1, c2=c2, t0=defaultLR)
            else:
                lr = linesearch(objGradFunc, d, grad, loss, gTd, pfixed_list, c1=c1, c2=c2,
                    t0=defaultLR)
        elif self.ls_type == "armijo":
            f = loss.item()
            if self.Ck==float('inf'):
                self.Ck=f
            if gTd > 0 and not self.pos_only:
                lr, f_new = armijo(objFunc, d.neg(), grad, max(self.Ck, f), defaultLR,
                    pfixed_list, -gTd, c1=c1)
                lr *= -1.0
            else:
                lr, f_new = armijo(objFunc, d, grad, max(self.Ck, f), defaultLR,
                    pfixed_list, gTd, c1=c1)
            #print(f"armijo: lr={lr}")
            self.Ck = self.monotone_strength* self.Qk* self.Ck + f_new
            self.Qk = self.monotone_strength* self.Qk + 1
            self.Ck/= self.Qk
        else:
            lr = defaultLR
            print(f"__stepsize_no_per_layer: Unrecognized lsType {self.ls_type}. Using default LR={lr}.")
            
        return lr, gTd

    @torch.no_grad()
    def step(self, closure, delay_start_step=0, ls_extra=False, normalize_directions=False):
        if ls_extra:
            raise NotImplementedError("Extra line-search optimizer groups are not supported")
        if len(self.ls_opt.param_groups) != 1:
            raise NotImplementedError("Wolfe line search currently supports one line-search parameter group")

        closure = torch.enable_grad()(closure)
        loss = closure()
        group = self.ls_opt.param_groups[0]
        pList = group['params']
        if not pList:
            raise ValueError("Line-search optimizer must contain at least one parameter")
        do_line_search = self.step_count >= delay_start_step

        if self.per_parameter:
            for _ in range(self.max_iter):
                grad = self._gather_flat_grad(pList)
                direction = self.get_direction(pList, group)
                # Search each parameter from the same base point, then apply all
                # accepted steps together so behavior does not depend on order.
                updates = []
                offset = 0
                for p in pList:
                    state = self.ls_opt.state[p]
                    self.__set_state_defaults(state)
                    numel = p.numel()
                    d = direction[offset : offset + numel]
                    p_grad = grad[offset : offset + numel]
                    offset += numel
                    if normalize_directions:
                        norm = (d**2).sum()**0.5
                        if norm > 1e-4:
                            d = d / norm
                    defaultLR = group['lr']
                    if do_line_search:
                        lr, gTd = self.__stepsize(
                            p, state['numel_cache'], closure, d, p_grad,
                            loss, self.c1, self.c2, defaultLR, update_reference=False,
                        )
                    else:
                        lr = defaultLR
                        gTd = torch.dot(d, p_grad)
                    self.__update_state(state, loss, lr, d, p_grad, gTd)
                    updates.append((p, state, lr, d))

                for p, state, lr, d in updates:
                    self._add_grad([p], state['numel_cache'], lr, d)
                loss = closure()
                if do_line_search and self.ls_type == "strong_wolfe":
                    self.__update_nonmonotone_reference(loss)
        else:
            self.__set_group_defaults(group)
            for _ in range(self.max_iter):
                grad = self._gather_flat_grad(pList)
                d = self.get_direction(pList, group)
                if normalize_directions:
                    norm = (d**2).sum()**0.5
                    if norm > 1e-4:
                        d /= norm
                defaultLR = group['lr']
                if do_line_search:
                    lr, gTd = self.__stepsize_no_per_layer(
                        pList, group['numel_cache'], closure, d, grad,
                        loss, self.c1, self.c2, defaultLR,
                    )
                else:
                    lr = defaultLR
                    gTd = torch.dot(d, grad)
                self.__update_group(group, loss, lr, d, grad, gTd)
                self._add_grad(pList, group['numel_cache'], lr, d)
                loss = closure()

        self.step_count += 1
        # Reserved for an inner-loop convergence test that is not implemented.
        return loss, False

    def step_extra(self, closure, delay_start_step=0):
        return self.step(closure, delay_start_step=delay_start_step, ls_extra=True)
