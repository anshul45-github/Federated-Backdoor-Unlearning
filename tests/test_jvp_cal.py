import torch
from src.unlearning.jvp_cal import ggn_vector_product, exact_hvp

def test_ggn_matches_explicit_jacobian():
    torch.manual_seed(1); m=torch.nn.Linear(2,3,bias=True).double(); x=torch.randn(2,2,dtype=torch.double)
    p=dict(m.named_parameters()); b=dict(m.named_buffers()); v={k:torch.randn_like(t) for k,t in p.items()}
    hv=ggn_vector_product(m,p,b,x,v)
    flatp=torch.cat([t.flatten() for t in p.values()]); flatv=torch.cat([t.flatten() for t in v.values()])
    def f(z):
        pos=0; pp={}
        for k,t in p.items(): pp[k]=z[pos:pos+t.numel()].view_as(t); pos+=t.numel()
        return torch.func.functional_call(m,(pp,b),(x,))
    J=torch.autograd.functional.jacobian(f,flatp).reshape(-1,flatp.numel())
    probs=f(flatp).softmax(-1); H=torch.zeros_like(J.T@J)
    for i in range(len(x)):
        q=torch.diag(probs[i])-probs[i:i+1].T@probs[i:i+1]
        Ji=J[i*3:(i+1)*3]; H+=Ji.T@q@Ji/len(x)
    got=torch.cat([z.flatten() for z in hv.values()])
    assert torch.allclose(got,H@flatv,atol=1e-9,rtol=1e-7)

def test_exact_hvp_finite_difference():
    torch.manual_seed(2); m=torch.nn.Linear(2,2).double(); x=torch.randn(3,2,dtype=torch.double); y=torch.tensor([0,1,1])
    p=dict(m.named_parameters()); b=dict(m.named_buffers()); v={k:torch.randn_like(t) for k,t in p.items()}
    hv=exact_hvp(m,p,b,x,y,v); eps=1e-5
    def grad(sign):
        pp={k:(t+sign*eps*v[k]).detach().requires_grad_() for k,t in p.items()}
        loss=torch.nn.functional.cross_entropy(torch.func.functional_call(m,(pp,b),(x,)),y)
        return torch.autograd.grad(loss,tuple(pp.values()))
    gp,gm=grad(1),grad(-1)
    for k,a,c in zip(p,gp,gm): assert torch.allclose(hv[k],(a-c)/(2*eps),atol=1e-7,rtol=1e-5)


def test_second_order_quadratic_two_steps():
    from src.unlearning.jvp_cal import taylor_update
    torch.manual_seed(3)
    A=torch.tensor([[2., .3],[.3,1.]],dtype=torch.double)
    old=torch.tensor([.2,-.4],dtype=torch.double)
    shift=torch.tensor([.1,.3],dtype=torch.double)
    eta=.02
    def update(x):
        z=x.clone()
        for _ in range(2): z=z-eta*(A@z)
        return z-x
    hist=update(old)
    hv=A@shift
    estimate=taylor_update({"x":hist},{"x":hv},{"x":A@hv},eta,2,2)["x"]
    assert torch.allclose(estimate,update(old+shift),atol=1e-12)

def test_real_model_ggn_smoke_and_buffers():
    from src.models.small_cnn import SmallCNN
    m=SmallCNN().eval()
    p=dict(m.named_parameters()); b={k:v.clone() for k,v in m.named_buffers()}
    original={k:v.clone() for k,v in b.items()}
    x=torch.randn(2,3,32,32); v={k:torch.randn_like(t) for k,t in p.items()}
    hv=ggn_vector_product(m,p,b,x,v)
    assert all(hv[k].shape==p[k].shape and hv[k].dtype==p[k].dtype and torch.isfinite(hv[k]).all() for k in p)
    assert all(torch.equal(b[k],original[k]) for k in b)
