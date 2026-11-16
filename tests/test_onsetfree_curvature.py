import torch

from src.unlearning.onsetfree.curvature import BenignCurvature, heavy_ball_contraction, lanczos_top, l2_tail, spectral_response
from src.models.small_cnn import SmallCNN


def test_ggn_matches_explicit_linear_softmax_matrix():
    model = torch.nn.Linear(2, 2, bias=False).eval()
    x = torch.tensor([[1.0, 2.0], [0.3, -1.0]])
    y = torch.tensor([0, 1])
    operator = BenignCurvature(model, [(x, y)])
    vector = torch.tensor([0.2, -0.4, 0.1, 0.3])
    weight = model.weight.detach().clone().requires_grad_()
    jacobian = torch.autograd.functional.jacobian(lambda w: torch.nn.functional.linear(x, w), weight).reshape(2, 2, 4)
    logits = torch.nn.functional.linear(x, weight)
    p = logits.softmax(dim=1)
    explicit = torch.zeros(4, 4)
    for i in range(2):
        h = torch.diag(p[i]) - torch.outer(p[i], p[i])
        explicit += jacobian[i].T @ h @ jacobian[i] / 2
    assert torch.allclose(operator.ggn(vector), explicit @ vector, atol=1e-6)


def test_curvature_snapshots_attack_parameters():
    model = torch.nn.Linear(2, 2, bias=False).eval()
    x = torch.tensor([[1.0, 2.0]])
    operator = BenignCurvature(model, [(x, torch.tensor([0]))])
    direction = torch.tensor([0.2, -0.4, 0.1, 0.3])
    before = operator.ggn(direction)
    with torch.no_grad():
        model.weight[0].add_(5.0)
    assert torch.allclose(operator.ggn(direction), before)


def test_hessian_matches_finite_difference():
    model = torch.nn.Linear(2, 2, bias=False).eval()
    x = torch.tensor([[1.0, 2.0], [0.3, -1.0]])
    y = torch.tensor([0, 1])
    vector = torch.tensor([0.2, -0.4, 0.1, 0.3])
    operator = BenignCurvature(model, [(x, y)])
    original = model.weight.detach().reshape(-1)

    def gradient(flat):
        w = flat.reshape(2, 2).detach().requires_grad_()
        loss = torch.nn.functional.cross_entropy(torch.nn.functional.linear(x, w), y)
        return torch.autograd.grad(loss, w)[0].reshape(-1)

    eps = 1e-3
    finite = (gradient(original + eps * vector) - gradient(original - eps * vector)) / (2 * eps)
    assert torch.allclose(operator.hessian(vector), finite, atol=1e-4)


def test_heavy_ball_and_lanczos_on_quadratic():
    h = torch.diag(torch.tensor([1.0, 3.0, 5.0]))
    values, basis = lanczos_top(lambda v: h @ v, 3, 2, torch.device("cpu"), iterations=3)
    assert torch.allclose(values, torch.tensor([5.0, 3.0]), atol=1e-5)
    assert torch.allclose((h @ basis).norm(dim=0), values, atol=1e-5)
    lam = torch.tensor([1.0])
    assert torch.allclose(heavy_ball_contraction(lam, 0.1, 0.0, 2), torch.tensor([0.81]))
    # With momentum 0.5, two steps yield 1 - 0.1 - 0.1*(0.5+0.9) = 0.76.
    assert torch.allclose(heavy_ball_contraction(lam, 0.1, 0.5, 2), torch.tensor([0.76]))


def test_spectral_response_matches_linear_quadratic_recursion():
    basis = torch.tensor([[0.0], [1.0]])
    rho = torch.tensor([0.5])
    forces = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    exact = torch.zeros(2)
    transition = torch.diag(torch.tensor([1.0, 0.5]))
    for force in forces:
        exact = transition @ exact - force
    assert torch.allclose(spectral_response(forces, basis, rho), exact)


def test_ggn_runs_on_smallcnn_with_eval_batchnorm():
    model = SmallCNN().eval()
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    operator = BenignCurvature(model, [(x, y)])
    width = sum(parameter.numel() for parameter in model.parameters())
    result = operator.ggn(torch.randn(width))
    assert result.shape == (width,)
    assert torch.isfinite(result).all()


def test_ggn_train_batchnorm_uses_batch_stats_without_mutating_buffers():
    model = SmallCNN().eval()
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    buffers_before = {name: value.clone() for name, value in model.named_buffers()}
    operator = BenignCurvature(model, [(x, y)], bn_mode="train")
    width = sum(parameter.numel() for parameter in model.parameters())
    result = operator.ggn(torch.randn(width))
    assert torch.isfinite(result).all()
    assert all(torch.equal(value, buffers_before[name]) for name, value in model.named_buffers())


def test_l2_tail_matches_quadratic_local_training():
    h = torch.diag(torch.tensor([0.0, 2.0]))
    initial = torch.tensor([1.0, 1.0])
    force = torch.tensor([0.1, 0.2])
    for steps, second_order in ((1, False), (2, True)):
        exact = initial.clone()
        for _ in range(steps):
            exact = exact - 0.1 * (h @ exact)
        exact -= force
        actual = l2_tail(initial, [force], [lambda v: h @ v], 0.1, steps, second_order)
        assert torch.allclose(actual, exact, atol=1e-7)
