"""Matrix-free benign curvature operators and momentum-aware local response."""

import torch
from torch.func import functional_call, jvp, vjp


def parameter_layout(model):
    return [(name, tuple(value.shape), value.numel()) for name, value in model.named_parameters()]


def unflatten(vector, layout):
    result = {}
    offset = 0
    for name, shape, width in layout:
        result[name] = vector[offset:offset + width].reshape(shape)
        offset += width
    if offset != vector.numel():
        raise ValueError("Vector length does not match model parameters")
    return result


def flatten(named, layout):
    return torch.cat([named[name].reshape(-1) for name, _, _ in layout])


class BenignCurvature:
    """Fixed-subsample GGN or exact-Hessian vector product with explicit BN mode."""

    def __init__(self, model, batches, device=None, bn_mode="eval"):
        if bn_mode not in ("eval", "train"):
            raise ValueError("bn_mode must be eval or train")
        self.model = model.eval()
        self.bn_mode = bn_mode
        if bn_mode == "train":
            for module in self.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.train()
        self.device = device or next(model.parameters()).device
        self.layout = parameter_layout(model)
        self.params = {name: value.detach().to(self.device).clone() for name, value in model.named_parameters()}
        self.buffers = {name: value.detach().to(self.device).clone() for name, value in model.named_buffers()}
        self.batches = [(x.to(self.device), y.to(self.device)) for x, y in batches]
        self.total = sum(len(y) for _, y in self.batches)
        if self.total <= 0:
            raise ValueError("Curvature subsample is empty")
        self.calls = 0

    def _logits(self, params, x):
        buffers = {name: value.clone() for name, value in self.buffers.items()} if self.bn_mode == "train" else self.buffers
        return functional_call(self.model, (params, buffers), (x,))

    def ggn(self, vector):
        direction = unflatten(vector, self.layout)
        result = torch.zeros_like(vector)
        for x, _ in self.batches:
            def logits(params):
                return self._logits(params, x)

            output, jv = jvp(logits, (self.params,), (direction,))
            probabilities = output.softmax(dim=1)
            weighted = probabilities * (jv - (probabilities * jv).sum(dim=1, keepdim=True))
            _, pullback = vjp(logits, self.params)
            result += flatten(pullback(weighted)[0], self.layout).detach() / self.total
        self.calls += 1
        return result

    def hessian(self, vector):
        params = {name: value.detach().requires_grad_() for name, value in self.params.items()}
        result = torch.zeros_like(vector)
        for x, y in self.batches:
            logits = self._logits(params, x)
            loss = torch.nn.functional.cross_entropy(logits, y, reduction="sum") / self.total
            grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=True)
            first = torch.cat([grad.reshape(-1) for grad in grads])
            second = torch.autograd.grad(torch.dot(first, vector), tuple(params.values()))
            result += torch.cat([tensor.reshape(-1) for tensor in second]).detach()
        self.calls += 1
        return result


def heavy_ball_contraction(lambdas, learning_rate, momentum, steps):
    """Exact scalar perturbation transfer for K constant-curvature SGD steps.

    Each client starts with zero optimizer momentum. This transfer replaces the
    no-momentum (1 - eta*lambda)^K heuristic used in the proposal.
    """
    displacement = torch.ones_like(lambdas)
    velocity = torch.zeros_like(lambdas)
    for _ in range(steps):
        velocity = momentum * velocity + lambdas * displacement
        displacement = displacement - learning_rate * velocity
    return displacement.clamp(0, 1)


def spectral_response(forces, basis, contractions):
    """Finite-horizon L1 response to ordered, already weighted contrasts.

    `forces[t]` is p_m,t * (U_m,t - B_t). Flat components persist; components
    in `basis` contract by `contractions` each subsequent round.
    """
    if not forces:
        raise ValueError("At least one round is required")
    total = torch.stack(forces).sum(dim=0)
    if basis.numel() == 0:
        return -total
    flat = total - basis @ (basis.T @ total)
    stiff = torch.zeros(basis.shape[1], device=total.device, dtype=total.dtype)
    horizon = len(forces)
    for index, force in enumerate(forces):
        stiff += contractions.pow(horizon - index - 1) * (basis.T @ force)
    return -flat - basis @ stiff


def l2_tail(initial, forces, hvps, learning_rate, steps, second_order=False):
    """Apply the registered one/two-HVP Taylor recursion over the tail.

    `hvps` contains a curvature operator per tail round. For a quadratic and
    one local step this is exact. The optional second-order term is exact for
    two gradient-descent steps on a fixed quadratic with zero momentum.
    """
    if len(forces) != len(hvps):
        raise ValueError("Need one force and curvature operator per round")
    delta = initial.clone()
    for force, hvp in zip(forces, hvps):
        first = hvp(delta)
        next_delta = delta - learning_rate * steps * first - force
        if second_order:
            next_delta += 0.5 * steps * (steps - 1) * learning_rate**2 * hvp(first)
        delta = next_delta
    return delta


def lanczos_top(matvec, dimension, rank, device, seed=42, iterations=None, dtype=torch.float32):
    """Full-reorthogonalized Lanczos, returning descending Ritz eigenpairs."""
    if rank <= 0 or dimension <= 0:
        raise ValueError("Positive rank and dimension required")
    steps = min(dimension, iterations or max(2 * rank, rank + 20))
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(dimension, device=device, dtype=dtype, generator=generator)
    q = q / q.norm()
    basis = []
    alphas = []
    betas = []
    previous = torch.zeros_like(q)
    previous_beta = 0.0
    for _ in range(steps):
        z = matvec(q) - previous_beta * previous
        alpha = torch.dot(q, z)
        z = z - alpha * q
        for old in basis:
            z -= torch.dot(old, z) * old
        beta = z.norm()
        basis.append(q)
        alphas.append(alpha)
        if beta.item() < 1e-8 or len(basis) == steps:
            break
        betas.append(beta)
        previous, q = q, z / beta
        previous_beta = beta
    diagonal = torch.stack(alphas)
    tridiagonal = torch.diag(diagonal)
    if betas:
        off_diagonal = torch.stack(betas)
        tridiagonal += torch.diag(off_diagonal, 1) + torch.diag(off_diagonal, -1)
    values, small_vectors = torch.linalg.eigh(tridiagonal)
    order = torch.argsort(values, descending=True)[:rank]
    large_vectors = torch.stack(basis, dim=1) @ small_vectors[:, order]
    return values[order], large_vectors
