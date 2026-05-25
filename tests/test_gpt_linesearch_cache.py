import torch

from nanochat.gpt import GPT, GPTConfig


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_model(n_layer=6):
    torch.manual_seed(1234)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=64,
        n_layer=n_layer,
        n_head=1,
        n_kv_head=1,
        n_embd=32,
        window_pattern="L",
    )
    model = GPT(config).to(DEVICE)
    model.init_weights()
    with torch.no_grad():
        for block in model.transformer.h:
            torch.nn.init.normal_(block.attn.c_proj.weight, std=0.02)
            torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.02)
    return model


def _inputs():
    torch.manual_seed(5678)
    idx = torch.randint(0, 64, (2, 8), device=DEVICE)
    targets = torch.randint(0, 64, (2, 8), device=DEVICE)
    return idx, targets


def _assert_cached_forward_matches_full_forward(fixed_block_ratio, expected_split_idx, n_layer=6):
    model = _make_model(n_layer=n_layer)
    idx, targets = _inputs()

    logits = model(idx)
    loss = model(idx, targets)
    with torch.no_grad():
        cache = model.build_linesearch_cache(
            idx, partial_linesearch_type=1, fixed_block_ratio=fixed_block_ratio
        )
    cached_logits = model.forward_from_linesearch_cache(cache)
    cached_loss = model.forward_from_linesearch_cache(cache, targets=targets)

    assert cache["split_idx"] == expected_split_idx
    torch.testing.assert_close(cached_logits, logits)
    torch.testing.assert_close(cached_loss, loss)


def test_type1_cache_matches_full_forward_when_split_precedes_backout():
    _assert_cached_forward_matches_full_forward(fixed_block_ratio=2 / 3, expected_split_idx=2)


def test_type1_cache_matches_full_forward_when_split_follows_backout():
    _assert_cached_forward_matches_full_forward(fixed_block_ratio=0.2, expected_split_idx=8, n_layer=10)


def test_type1_cache_matches_full_forward_gradients_for_linesearch_parameters():
    model = _make_model(n_layer=10)
    idx, targets = _inputs()
    fixed_block_ratio = 0.2
    linesearch_groups, _ = model._get_params_dict_opt_wrapper(
        partial_linesearch_type=1, fixed_block_ratio=fixed_block_ratio
    )
    linesearch_params = linesearch_groups[0]["params"]

    model.zero_grad(set_to_none=True)
    model(idx, targets).backward()
    full_grads = [param.grad.detach().clone() for param in linesearch_params]

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        cache = model.build_linesearch_cache(
            idx, partial_linesearch_type=1, fixed_block_ratio=fixed_block_ratio
        )
    model.forward_from_linesearch_cache(cache, targets=targets).backward()

    for param, full_grad in zip(linesearch_params, full_grads):
        torch.testing.assert_close(param.grad, full_grad)
