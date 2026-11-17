"""Matrix-free local-update Taylor estimators for the JVP-Cal experiment."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp, vjp


def flatten(params):
    return torch.cat([x.reshape(-1) for x in params.values()])

def unflatten_like(vector, params):
    out, pos = {}, 0
    for k, p in params.items():
        n=p.numel(); out[k]=vector[pos:pos+n].view_as(p); pos += n
    if pos != vector.numel(): raise ValueError("vector length mismatch")
    return out

def logits_fn(model, params, buffers, x):
    return functional_call(model, (params, buffers), (x,))

def ggn_vector_product(model, params, buffers, x, vector):
    """Empirical softmax GGN of mean cross entropy, using JVP and VJP."""
    def f(p): return logits_fn(model,p,buffers,x)
    _, jv = jvp(f, (params,), (vector,))
    logits=f(params).detach(); probs=logits.softmax(-1)
    lam_jv=probs*(jv-(probs*jv).sum(-1,keepdim=True))
    _, pullback=vjp(f, params)
    return pullback(lam_jv / x.shape[0])[0]

def exact_hvp(model, params, buffers, x, y, vector):
    p={k:v.detach().requires_grad_(True) for k,v in params.items()}
    loss=F.cross_entropy(logits_fn(model,p,buffers,x),y)
    grads=torch.autograd.grad(loss, tuple(p.values()), create_graph=True)
    dot=sum((g*v).sum() for g,v in zip(grads,vector.values()))
    hv=torch.autograd.grad(dot, tuple(p.values()), allow_unused=True)
    return {k:(torch.zeros_like(v) if h is None else h.detach()) for (k,v),h in zip(p.items(),hv)}

def taylor_update(u, hv, hv2=None, eta=.01, steps=1, order=1):
    out={k:v-eta*steps*hv[k] for k,v in u.items()}
    if order>=2:
        if hv2 is None: raise ValueError("second order requires H(H Delta)")
        c=.5*steps*(steps-1)*eta*eta
        out={k:out[k]+c*hv2[k] for k in u}
    return out

def rescale_global(update, reference):
    keys=[k for k,v in update.items() if v.is_floating_point() and "running_" not in k and "num_batches_tracked" not in k]
    a=torch.sqrt(sum(update[k].float().square().sum() for k in keys)); b=torch.sqrt(sum(reference[k].float().square().sum() for k in keys))
    s=b/a.clamp_min(1e-12)
    return {k:(update[k]*s if k in keys else update[k].clone()) for k in update}
