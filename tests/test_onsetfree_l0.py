import torch

from src.unlearning.onsetfree.l0 import apply_update, make_updates, project_out


def test_project_out_orthogonal_and_bn_preserved():
    x = torch.tensor([3.0, 4.0])
    v = torch.tensor([[1.0], [0.0]])
    assert torch.allclose(project_out(x, v), torch.tensor([0.0, 4.0]))
    state = {"weight": x, "bn.running_mean": torch.tensor([7.0]), "bn.num_batches_tracked": torch.tensor(2)}
    result = apply_update(state, {"weight": -project_out(x, v)})
    assert torch.allclose(result["weight"], torch.tensor([3.0, 0.0]))
    assert result["bn.running_mean"].item() == 7.0
    assert result["bn.num_batches_tracked"].item() == 2


def test_contrast_only_and_subtraction_differ():
    contrasts = {"weight": [(0.25, torch.tensor([2.0, 0.0]))]}
    benign_rows = {"weight": [torch.tensor([1.0, 0.0])]}
    malicious_rows = {"weight": [torch.tensor([3.0, 0.0])]}
    attacker_sum = {"weight": torch.tensor([0.75, 0.0])}
    updates, _ = make_updates(contrasts, benign_rows, malicious_rows, attacker_sum, rank=1)
    assert torch.allclose(updates["c1"]["weight"], torch.tensor([-0.75, 0.0]))
    assert torch.allclose(updates["c2"]["weight"], torch.tensor([-0.5, 0.0]))
