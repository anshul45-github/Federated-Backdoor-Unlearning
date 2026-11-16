import torch

from src.training.fedavg import aggregate_fedavg
from src.unlearning.onsetfree.traj_io import Trajectory, reconstruction_error


def test_loader_reconstructs_float_and_integer_buffers(tmp_path):
    initial = {
        "weight": torch.tensor([1.0, -2.0]),
        "bn.running_mean": torch.tensor([0.2]),
        "bn.num_batches_tracked": torch.tensor(3, dtype=torch.long),
    }
    deltas = {
        0: {"weight": torch.tensor([0.5, 0.2]), "bn.running_mean": torch.tensor([0.3]), "bn.num_batches_tracked": torch.tensor(1.0)},
        1: {"weight": torch.tensor([-0.1, 0.4]), "bn.running_mean": torch.tensor([-0.2]), "bn.num_batches_tracked": torch.tensor(2.0)},
    }
    rd = tmp_path / "trajectory" / "round_001"
    rd.mkdir(parents=True)
    torch.save(initial, tmp_path / "initial_model.pt")
    for i in deltas:
        torch.save(deltas[i], rd / f"delta_client_{i:02d}.pt")
    torch.save(aggregate_fedavg(initial, [deltas[0], deltas[1]], [1, 3]), rd / "global_model.pt")

    record = next(iter(Trajectory(tmp_path, {0: 1, 1: 3})))
    assert record.weights == {0: 0.25, 1: 0.75}
    assert max(reconstruction_error(record).values()) < 1e-6
