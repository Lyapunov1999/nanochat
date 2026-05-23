import torch

from nanochat.optim import OptimizerWithLineSearch


def _make_wrapper(params, *, lr=0.1, momentum=0.0, per_parameter=False):
    ls_opt = torch.optim.SGD([{"params": params}], lr=lr, momentum=momentum)
    fixed_opt = torch.optim.SGD([{"params": []}], lr=lr)
    wrapper = OptimizerWithLineSearch(
        ls_opt,
        fixed_opt,
        per_parameter=per_parameter,
        ls_type="strong_wolfe",
    )
    return wrapper, ls_opt


def test_group_direction_generation_does_not_recompute_closure():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    wrapper, ls_opt = _make_wrapper([p])
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        ls_opt.zero_grad()
        loss = 0.5 * p.square().sum()
        loss.backward()
        return loss

    _, all_done = wrapper.step(closure, delay_start_step=1)

    torch.testing.assert_close(p, torch.tensor([0.9]))
    assert calls == 2
    assert all_done is False


def test_per_parameter_delayed_step_does_not_leak_updates():
    p1 = torch.nn.Parameter(torch.tensor([1.0]))
    p2 = torch.nn.Parameter(torch.tensor([2.0]))
    wrapper, ls_opt = _make_wrapper([p1, p2], per_parameter=True)
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        ls_opt.zero_grad()
        loss = 0.5 * (p1.square() + p2.square()).sum()
        loss.backward()
        return loss

    wrapper.step(closure, delay_start_step=1)

    torch.testing.assert_close(p1, torch.tensor([0.9]))
    torch.testing.assert_close(p2, torch.tensor([1.8]))
    assert calls == 2


def _run_per_parameter_order(order):
    left = torch.nn.Parameter(torch.tensor([0.8]))
    right = torch.nn.Parameter(torch.tensor([-0.3]))
    params = {"left": left, "right": right}
    wrapper, ls_opt = _make_wrapper([params[name] for name in order], lr=0.2, per_parameter=True)

    def closure():
        ls_opt.zero_grad()
        residual = left + 2.0 * right - 1.2
        loss = 0.5 * (residual.square() + 0.2 * left.square() + 0.2 * right.square()).sum()
        loss.backward()
        return loss

    wrapper.step(closure)
    return left.detach(), right.detach()


def test_per_parameter_wolfe_search_is_independent_of_parameter_order():
    forward = _run_per_parameter_order(["left", "right"])
    reverse = _run_per_parameter_order(["right", "left"])

    torch.testing.assert_close(forward[0], reverse[0])
    torch.testing.assert_close(forward[1], reverse[1])


def test_multiple_line_search_parameter_groups_are_rejected():
    p1 = torch.nn.Parameter(torch.tensor([1.0]))
    p2 = torch.nn.Parameter(torch.tensor([2.0]))
    ls_opt = torch.optim.SGD([{"params": [p1]}, {"params": [p2]}], lr=0.1)
    fixed_opt = torch.optim.SGD([{"params": []}], lr=0.1)
    wrapper = OptimizerWithLineSearch(ls_opt, fixed_opt)

    def closure():
        ls_opt.zero_grad()
        loss = 0.5 * (p1.square() + p2.square()).sum()
        loss.backward()
        return loss

    try:
        wrapper.step(closure)
    except NotImplementedError:
        return
    raise AssertionError("Expected multiple line-search groups to be rejected")


def test_negative_wolfe_step_handles_momentum_ascent_direction():
    for per_parameter in (False, True):
        p = torch.nn.Parameter(torch.tensor([-0.1]))
        wrapper, ls_opt = _make_wrapper([p], momentum=0.9, per_parameter=per_parameter)
        ls_opt.state[p]["momentum_buffer"] = torch.tensor([1.0])

        def closure():
            ls_opt.zero_grad()
            loss = 0.5 * p.square().sum()
            loss.backward()
            return loss

        initial_loss = closure().item()
        wrapper.step(closure)
        final_loss = closure().item()
        prev_lr = ls_opt.state[p]["prev_lr"] if per_parameter else ls_opt.param_groups[0]["prev_lr"]

        assert final_loss < initial_loss
        assert prev_lr < 0


def test_delay_start_step_and_checkpoint_state_use_outer_step_count():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    wrapper, ls_opt = _make_wrapper([p], lr=0.2)

    def closure():
        ls_opt.zero_grad()
        loss = 5.0 * p.square().sum()
        loss.backward()
        return loss

    wrapper.step(closure, delay_start_step=2)
    torch.testing.assert_close(p, torch.tensor([-1.0]))
    assert wrapper.step_count == 1

    saved = wrapper.state_dict()
    resumed_p = torch.nn.Parameter(p.detach().clone())
    resumed_wrapper, resumed_opt = _make_wrapper([resumed_p], lr=0.2)
    resumed_wrapper.load_state_dict(saved)

    def resumed_closure():
        resumed_opt.zero_grad()
        loss = 5.0 * resumed_p.square().sum()
        loss.backward()
        return loss

    resumed_wrapper.step(resumed_closure, delay_start_step=2)
    torch.testing.assert_close(resumed_p, torch.tensor([1.0]))
    resumed_wrapper.step(resumed_closure, delay_start_step=2)
    torch.testing.assert_close(resumed_p, torch.tensor([0.0]), atol=1e-6, rtol=0.0)
    assert resumed_wrapper.step_count == 3

    immediate_p = torch.nn.Parameter(torch.tensor([1.0]))
    immediate_wrapper, immediate_opt = _make_wrapper([immediate_p], lr=0.2)

    def immediate_closure():
        immediate_opt.zero_grad()
        loss = 5.0 * immediate_p.square().sum()
        loss.backward()
        return loss

    immediate_wrapper.step(immediate_closure, delay_start_step=0)
    torch.testing.assert_close(immediate_p, torch.tensor([0.0]), atol=1e-6, rtol=0.0)


def test_legacy_checkpoint_step_is_converted_to_outer_step_count():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    wrapper, _ = _make_wrapper([p])
    saved = wrapper.state_dict()
    del saved["wrapper_state"]["step_count"]
    saved["ls_opt"]["state"][0] = {"step": 2}

    resumed_p = torch.nn.Parameter(torch.tensor([1.0]))
    resumed_wrapper, _ = _make_wrapper([resumed_p])
    resumed_wrapper.load_state_dict(saved)

    assert resumed_wrapper.step_count == 3
