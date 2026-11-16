from src.unlearning.onsetfree.onset import calibrate_from_benign, cusum, robust_z


def test_cusum_detects_shift_with_benign_calibrated_threshold():
    benign = [{i: float(i % 3) for i in range(9)} for _ in range(20)]
    threshold, peaks = calibrate_from_benign(benign, kappa=1.0)
    assert all(peak < threshold for peak in peaks.values())
    z = [robust_z(1.0 if t < 10 else 10.0, list(round_values.values())) for t, round_values in enumerate(benign)]
    trace, onset = cusum(z, 1.0, threshold)
    assert onset == 11
    assert trace[9] < threshold < trace[10]
