import torch
#from torch.nn.utils.stateless import functional_call
from torch.func import functional_call, grad
import copy
import math

# ported from torch.optim.lbfgs which ported from https://github.com/torch/optim/blob/master/polyinterp.lua
def cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    if abs(x1-x2) < 1e-9:
        return (x1 + x2) / 2.0

    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0

# ported from torch.optim.lbfgs which ported from https://github.com/torch/optim/blob/master/lswolfe.lua
def strong_wolfe(obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9,
 max_ls=50, min_t=-float('inf'), max_t=float('inf'), non_monotone_f=None,
 trace_performance=False, mult=2.0, use_cubic_interp=True):
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    if non_monotone_f is None:
        non_monotone_f = f
    non_monotone_f = max(f, non_monotone_f)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        if f_new > (non_monotone_f + c1 * t * gtd) or (ls_iter > 1 and f_new > f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            if trace_performance:
                if f_new > (f + c1 * t * gtd):
                    print(f" bracket is [{t_prev}, {t}] because armijo no longer satisfied")
                else:
                    print(f" bracket is [{t_prev}, {t}] because we increased the function value {f_prev} --> {f_new}")
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            if trace_performance:
                print(f" done=True without zoom t={t}")
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            if trace_performance:
                print(f" bracket is [{t_prev}, {t}] because gtd_new is positive")
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        if min_step < min_t:
            min_step = min_t
        max_step = t * 10
        if max_step > max_t:
            max_step = max_t
        tmp = t
        if use_cubic_interp:
            t = cubic_interpolate(
                t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step))
        else:
            t *= mult
        #print(f"  bracket, trial t={t:>0.4f}, t_prev={t_prev:>0.4f}, t={tmp:>0.4f}")

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        # reached max number of iterations?
        if ls_iter == max_ls:
            bracket = [0, t]
            bracket_f = [f, f_new]
            bracket_g = [g, g_new]

        # t_prev was within min_t and max_t but t is not?
        if t <= min_t or t >= max_t: # should just need equality check
            # this is not the best bracket but just the safest to 
            #  pass to zoom
            bracket = [min_t, max_t]
            f_min, g_min = obj_func(x, min_t, d)
            f_max, g_max = obj_func(x, max_t, d)
            ls_func_evals += 2 # TODO: can at least reduce this by 1
            bracket_f = [f_min, f_max]
            bracket_g = [g_min, g_max]
            bracket_gtd = [g_min.dot(d), g_max.dot(d)]
            break

    # zoom phase
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)  # type: ignore[possibly-undefined]
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:  # type: ignore[possibly-undefined]
            if trace_performance:
                print(f"  break on zoom: {abs(bracket[1]-bracket[0])*d_norm} < {tolerance_change}, t=({bracket[0]}, {bracket[1]})")
            break

        # compute new trial value
        if use_cubic_interp:
            t = cubic_interpolate(
                bracket[0],
                bracket_f[0],
                bracket_gtd[0],  # type: ignore[possibly-undefined]
                bracket[1],
                bracket_f[1],
                bracket_gtd[1])
        else:
            t = (bracket[0]+bracket[1])/mult
        if trace_performance:
            print(f"  zoom: trial t={t:>0.4f}, bracket[0]={bracket[0]:>0.4f}, bracket[1]={bracket[1]:>0.4f}")
        
        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (non_monotone_f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]  # type: ignore[possibly-undefined]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_gtd[low_pos] = gtd_new
        
    # return stuff
    t = bracket[low_pos]  # type: ignore[possibly-undefined]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]  # type: ignore[possibly-undefined]
    if trace_performance:
        print(f"strong_wolfe returns t={t}, f_new={f_new} after {ls_func_evals} evals")
    return f_new, g_new, t, ls_func_evals

# another implementation of strong Wolfe conditions
# works whether p0 is a list or a Tensor as long as the objGradFunc matches
def linesearch(objGradFunc, d, g0, f0, gTd0, p0, c1=1e-4, c2=0.9, t0=1, mult=10.0, maxLsIter=25):
    nLsIter = 0
    eps = 1e-15
    alphaLo, alphaHi = eps, t0
    c1gTd0, c2gTd0 = c1*gTd0, c2*gTd0
    g, f, t, tprev, gTd = g0, f0, t0, eps, gTd0

    # bracketing phase
    while nLsIter < maxLsIter:
        gprev, fprev, gTdprev = g, f, gTd
        
        f, g = objGradFunc(p0, t, d)
        gTd = torch.dot(g,d)

        # condition 1 for exiting bracketing phase: 
        #   phi(alpha_i) > phi(0)+c1*alpha_i*phi'(0) OR [phi(alpha_i) >= phi(alpha_{i-1}) and i>1]
        thresh = f0 + t*c1gTd0
        if f > thresh or (nLsIter>0 and f >= fprev):
            alphaLo = tprev
            alphaHi = t
            #print(f"condition 1 satisfied at {nLsIter} iteration: lo = {alphaLo}, hi = {alphaHi}")
            break #zoom(tprev, t)

        # evaluate phi'(alpha_i)
        if abs(gTd) <= -c2gTd0:
            #print(f"found t={t} in bracketing phase using {nLsIter} iterations")
            return t # found it
        elif gTd >= 0:
            alphaLo = t
            alphaHi = tprev
            break #zoom(t, tprev)

        tprev = t
        t *= mult

        nLsIter += 1
    
    if nLsIter > maxLsIter:
        #print(f" spent all maxLsIter={maxLsIter} in bracketing phase and didn't find a good step size")
        return 1e-3 # failed so return a default step size
    
    # zoom phase
    foundIt = False
    while nLsIter <= maxLsIter:
        gprev, fprev, tprev, gTdprev = g, f, t, gTd
        t = (alphaLo + alphaHi)/2.0
        #print(f"  zoom({alphaLo:>0.4f}, {alphaHi:>0.4f}): t={t}")

        # evaluate phi(alpha_j)
        f, g = objGradFunc(p0, t, d)
        gTd = torch.dot(g, d)

        thresh = f0 + t*c1gTd0
        # condition 1
        fLo, _ = objGradFunc(p0, alphaLo, d)
        if f > thresh or f >= fLo:
            alphaHi = t
            #print(f"  zoom: f={f:>0.4f} > thresh={thresh:>0.4f} or > fLo={fLo:>0.4f}.")
        else:
            # evaluate phi'(alpha_j)
            if abs(gTd) <= -c2gTd0: # stopping condition
                foundIt = True
                break
            if gTd* (alphaHi-alphaLo) >= 0.0:
                #print(f"  zoom: gTd={gTd:>0.4f}, (alphaHi-alphaLo)={(alphaHi-alphaLo):>0.4f}")
                alphaHi = alphaLo
            alphaLo = t
        nLsIter += 1

        if abs(alphaHi-alphaLo) < eps: # bracket too small
            #print(f"  zoom: bracket too small: alphaHi={alphaHi:0.4f}, alphaLo={alphaLo:0.4f}")
            break
    
    if not foundIt:
        #print(f" spent maxLsIter={maxLsIter}. at zoom(alphaLo={alphaLo:>0.4f}, alphaHi={alphaHi:>0.4f}) and didn't find a good step size")
        return (alphaLo+alphaHi)/2.0 # failed so return something within zoom range

    #print(f" SUCCESS returning {t} in {nLsIter} iterations")
    return t

# Armijo: non-monotone if the f0 passed in is not the latest f0
def armijo(objFunc, d, g0, f0, t0, p0, gTd0, c1=1e-4, mult=0.5, maxLsIter=25, 
    non_monotone_f=None, trace_performance=False, use_mult=False):
    nLsIter = 0
    c1gTd = c1*gTd0

    t = t0
    f = objFunc(p0, t, d)
    if non_monotone_f is None:
        non_monotone_f = f0
    non_monotone_f = max(f0, non_monotone_f)
    thresh = non_monotone_f + t*c1gTd
    if trace_performance:
        print(f"armijo: f={f}, f0={f0}, non_monotone_f={non_monotone_f}, thresh={thresh}")
    while nLsIter < maxLsIter:
        if trace_performance:
            print(f"armijo: nLsIter={nLsIter}, t={t:>0.3f}, f={f:>0.3f}, thresh={thresh:>0.3f}")
        if f < thresh:
            if trace_performance:
                print(f"armijo returns successfully at nLsIter={nLsIter}: t={t}, f={f}, f0={f0}, non_monotone_f={non_monotone_f}")
            return t, f
        
        if use_mult:
            t *= mult
        else:
            t_prev = t
            t = -t_prev**2 * gTd0/(2*(f-f0-t_prev*gTd0))
            if t<0 or t > t_prev or math.isnan(t):
                t = t_prev * mult
        
        f = objFunc(p0, t, d)
        thresh = non_monotone_f + t*c1gTd
        nLsIter += 1
    if trace_performance:
        print(f"armijo fails and returns t={t}, f={f0}, f={f}")
    if isinstance(f, float):
        return 0, f0
    else:
        return 0, torch.ones(1).to(f.device) *f0 

# moves model parameters by stepsize * direction and stores the change in parameters in updates
@torch.no_grad()
def SO_perturb_parameters(num_d, model, updates, stepsizes, directions):
    #print(f"SO_perturb_parameters with stepsizes={stepsizes}")
    for name, p in model.named_parameters():
        if p.requires_grad:
            #print(f"before: {name}, {(p**2).sum()**0.5}")
            update = 0
            for a in range(num_d):
                update = update + stepsizes[a] * directions[name][a]
            p.add_(update)
            #print(f"after: {name}, {(p**2).sum()**0.5}")
            updates[name] = update

# subtracts the update amount store in updates from the model's parameters 
@torch.no_grad()
def SO_revert_parameters(model, updates):
    #print("SO_revert_parameters")
    for name, p in model.named_parameters():
        if p.requires_grad:
            #print(f"before: {name}, {(p**2).sum()**0.5}")
            p.subtract_(updates[name])
            #print(f"after: {name}, {(p**2).sum()**0.5}")
    

# gets function value that is L2-regularized by step size
def SO_get_regularized_loss(obj_func, reg_weight, stepsizes=None):
    loss = obj_func()
    if reg_weight > 0 and stepsizes is not None:
        loss += reg_weight * (stepsizes ** 2).sum()
    return loss

# returns loss and gradient wrt parameters if stepsize were to be moved by the stepsize-of-stepsize (lr_lr)
#  in the direction of the gradient of the step size (d_SO). 
def SO_try_step_in_direction(stepsizes, d_SO, lr_lr, num_d, model, directions, reg_weight, obj_func, trainable_params, calc_grad=False):
    updates = {}
    stepsizes.add_(d_SO, alpha=lr_lr)
    SO_perturb_parameters(num_d, model, updates, stepsizes, directions)
    temp_loss = SO_get_regularized_loss(obj_func, reg_weight, stepsizes)
    temp_grad = None
    if calc_grad:
        grads = torch.autograd.grad(temp_loss, trainable_params)
        grads_map = dict(zip(trainable_params, grads))
        temp_grad = torch.zeros(num_d).to(model.device)
        for b in range(num_d):
            for name, p in model.named_parameters():
                if p.requires_grad:
                    temp_grad[b] += (grads_map[p] * directions[name][b]).sum()
            if reg_weight > 0:
                temp_grad[b] += 2 * reg_weight * stepsizes[b] # l2 reg stepsizes
                # TODO: regularize update and not just weights
    SO_revert_parameters(model, updates)
    stepsizes.subtract_(d_SO, alpha=lr_lr)
    return temp_loss, temp_grad

# returns loss and gradient wrt step size, i.e. phi(alpha) and phi'(alpha)
# NOTE: Raydan's globalized BB method uses bb-long-step
def SO_get_loss_and_grad_of_stepsize(num_d, model, trainable_params, stepsizes, directions, reg_weight, obj_func,
 prev_grads_SO, stepsizes_prev, alpha, trace_performance=False, bb_use_short_step=True,
 delta=1.0):
    updates = {}
    # \theta = \theta + Dt
    SO_perturb_parameters(num_d, model, updates, stepsizes, directions)
    # dL/d\theta at the new perturbed point
    loss = SO_get_regularized_loss(obj_func, reg_weight, stepsizes)

    # calculate grads_SO by chain rule
    grads = torch.autograd.grad(loss, trainable_params)
    grad_map = dict(zip(trainable_params, grads))
    grads_SO = torch.zeros(num_d).to(model.device)
    for b in range(num_d):
        for name, p in model.named_parameters():
            if p.requires_grad:
                grads_SO[b] += (grad_map[p] * directions[name][b]).sum() #gTd = \nabla L(\theta+td)^Td when t=0
            
        if reg_weight > 0:
            grads_SO[b] += 2 * reg_weight * stepsizes[b] # l2 reg stepsizes
            # TODO: regularize update and not just weights
    
    # BB direction
    with torch.no_grad():
        bb_SO = torch.zeros(num_d).to(model.device)
        if prev_grads_SO is not None:
            for b in range(num_d):        
                diffG = grads_SO[b] - prev_grads_SO[b]
                diffX = stepsizes[b] - stepsizes_prev[b]
                if bb_use_short_step:
                    bb_mult = diffX* diffG/ (diffG*diffG) 
                else:
                    bb_mult = diffX*diffX/ (diffX*diffG)

                if trace_performance:
                    print(f"SO_get_loss_and_grad_of_stepsize: bb_mult={bb_mult}, diffG={(diffG**2).sum()**0.5}, diffX={(diffX**2).sum()**0.5}")
                if math.isnan(bb_mult) or math.isinf(bb_mult):
                    alpha[b] = delta
                elif bb_mult < 0:
                    alpha[b] = -bb_mult
                elif bb_mult < 1e-10:
                    alpha[b] = delta
                elif bb_mult > 1e10:
                    alpha[b] = delta
                    delta *= 2
                else:
                    alpha[b] = bb_mult
                if trace_performance:
                    print(f"SO_get_loss and grad of stepsize: alpha[{b}]={alpha[b]}")
                bb_SO[b] = -grads_SO[b] * alpha[b]
        else:
            for b in range(num_d):
                bb_SO[b] = -grads_SO[b]
    
    if trace_performance:
        print(f"SO_get_loss_and_grad_of_stepsize: grads_SO={grads_SO}, bb_SO={bb_SO}")
        for name, p in model.named_parameters():
            print(f"{name}: norm(updates) = {(updates[name]**2).sum()**0.5}")

    SO_revert_parameters(model, updates)
    return loss, grads_SO, bb_SO, alpha, delta
    
@torch.enable_grad()
def SO(num_d, directions, obj_func, model, t0=None, reg_weight=0.0, trace_performance=False,
    lr_lr=2, max_iter=250, opt_tol=1e-9, prog_tol=1e-18, use_bb=True, use_strong_wolfe=False):
    stepsize_reset_size = 1e-5
    SO_status = 0

    if t0 is not None:
        stepsizes = t0.detach().clone().requires_grad_(True)
    else:
        stepsizes = torch.ones(num_d, requires_grad=True) * stepsize_reset_size

    stepsizes = stepsizes.to(model.device)
    stepsizes_prev = None 

    # original loss before any step taken
    with torch.no_grad():
        orig_loss = SO_get_regularized_loss(obj_func, reg_weight, None)
    if trace_performance:
        print(f"SO start: num_d={num_d}, t0={stepsizes}, L0={orig_loss:.4f}")
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(f"  {name}: ",end="")
                for a in range(num_d):
                    print(f"norm dir {a}: {(directions[name][a]**2).sum()**0.5},  ", end="")
                print("")

    prev_loss, loss, prev_grads_SO = orig_loss, None, None
    alpha = torch.ones(num_d) # BB mult
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    delta = 1.0
    last_few_fs = [orig_loss.item()]

    for a in range(max_iter): # outer loop of inner loop
        initial_ls_loss, grads_SO, bb_SO, alpha, delta = SO_get_loss_and_grad_of_stepsize(num_d,  
            model, trainable_params, stepsizes, directions, reg_weight, obj_func, 
            prev_grads_SO, stepsizes_prev, alpha, trace_performance=trace_performance,
            delta=delta)    
        curr_loss = initial_ls_loss

        stepsizes_prev = stepsizes.clone().detach()

        if trace_performance:
            print(f"  SO {a}: stepsizes={stepsizes}, grads of stepsizes={grads_SO}")
        
        grads_SO_opt = grads_SO.abs().max()
        if grads_SO_opt < opt_tol:
            if trace_performance:
                print(f"  SO {a}: early return (opt_tol), grads_SO_opt={grads_SO_opt}, loss={loss:.4f}, stepsizes={stepsizes}")
            if curr_loss > orig_loss:
                if trace_performance:
                    print(f"  *** curr_loss={curr_loss} > orig_loss={orig_loss}, setting to small step size")
                stepsizes = torch.ones(num_d) * stepsize_reset_size
                SO_status = 1
            return SO_status, stepsizes.detach() 

        # line search on lr_lr within GD on lr
        with torch.no_grad():
            # initialize line search on lr_lr
            stepsizes = stepsizes.detach()
            max_inner_LS = 250 # the original optPS has this set to 250
            if use_bb:
                d_SO = bb_SO
            else:
                d_SO = -grads_SO

            gTd_SO = torch.dot(grads_SO, d_SO)
            if trace_performance:
                print(f" SO {a}: gTd_SO={gTd_SO}")
            if use_strong_wolfe:
                def obj_grad_func_SO(stepsizes, curr_lr_lr, d_SO):
                    with torch.enable_grad():
                        f_SO, g_SO = SO_try_step_in_direction(stepsizes, d_SO, curr_lr_lr, num_d, model, directions, reg_weight, obj_func, trainable_params, calc_grad=True)
                    return f_SO, g_SO
            
                curr_loss, g_new, curr_lr_lr, ls_func_evals = strong_wolfe( obj_grad_func_SO,
                    stepsizes, lr_lr, d_SO, curr_loss, grads_SO, gTd_SO, c2=0.9, max_ls=max_inner_LS,
                    non_monotone_f=curr_loss, trace_performance=trace_performance)

                if trace_performance:
                    print(f"strong wolfe returns f={curr_loss_ls} and lr_lr={curr_lr_lr} after {ls_func_evals} function evaluations. f_prev={prev_loss_ls}. diff f={prev_loss_ls-curr_loss_ls}")
                    if prev_loss_ls < curr_loss_ls:
                        print(f"strong wolfe fails: prev_loss_ls={prev_loss_ls}, curr_loss_ls={curr_loss_ls}")
            else:
                def obj_func_SO(stepsizes, curr_lr_lr, d_SO):
                    with torch.enable_grad():
                        f_SO, _ = SO_try_step_in_direction(stepsizes, d_SO, curr_lr_lr, num_d, model, directions, reg_weight, obj_func, trainable_params, calc_grad=True)
                    return f_SO
                 
                curr_lr_lr, curr_loss = armijo(obj_func_SO, d_SO, grads_SO, curr_loss, 
                    lr_lr, stepsizes, gTd_SO, maxLsIter=max_inner_LS, non_monotone_f=max(last_few_fs), 
                    trace_performance=trace_performance)
                
                if len(last_few_fs) > 4:
                    last_few_fs.pop(0)
                last_few_fs.append(curr_loss.item())

                if trace_performance:
                    print(f"armijo returns f={curr_loss} and lr_lr={curr_lr_lr}. f_prev={prev_loss}. diff f={prev_loss-curr_loss}")

        stepsizes.add_(d_SO, alpha=curr_lr_lr)
        stepsizes.requires_grad_(True)
        
        if trace_performance:
            print(f"  SO {a}: linesearch sets stepsize={stepsizes}")

        if stepsizes_prev is not None:        
            stepsizes_prog = ((stepsizes/stepsizes_prev)-1).abs().max()
            if stepsizes_prog < prog_tol:
                if trace_performance:
                    print(f"  SO {a}: early return (prog_tol), prog={stepsizes_prog}, loss={curr_loss}, stepsizes={stepsizes}")
                if curr_loss > orig_loss:
                    if trace_performance:
                        print(f"  *** curr_loss={curr_loss} > orig_loss={orig_loss}, setting to small step size")
                    stepsizes = torch.ones(num_d) * stepsize_reset_size
                    SO_status = 1
                return SO_status, stepsizes.detach()

        prev_loss = curr_loss
        prev_grads_SO = grads_SO.clone().detach()   

    if trace_performance:
        print(f"SO end. returning {stepsizes}. original loss={orig_loss}, new loss={curr_loss}")
    if curr_loss > orig_loss:
        if trace_performance:
            print(f"  *** curr_loss={curr_loss} > orig_loss={orig_loss}, setting to small step size")
        stepsizes = torch.ones(num_d) * stepsize_reset_size
        SO_status = 1

    return SO_status, stepsizes.detach()