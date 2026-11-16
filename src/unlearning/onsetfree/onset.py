"""Trigger-free robust change-point statistics for stored client updates."""

import statistics


def robust_z(target, null, epsilon=1e-8):
    center = statistics.median(null)
    mad = statistics.median(abs(value - center) for value in null)
    return (target - center) / (1.4826 * mad + epsilon)


def cusum(z_scores, kappa, threshold=None):
    values = []
    current = 0.0
    first_alarm = None
    for round_number, z in enumerate(z_scores, start=1):
        current = max(0.0, current + z - kappa)
        values.append(current)
        if threshold is not None and first_alarm is None and current > threshold:
            first_alarm = round_number
    return values, first_alarm


def calibrate_from_benign(benign_residuals, kappa, epsilon=1e-6):
    """Conservative <=5% empirical false alarms with nine benign clients.

    Nine pseudo-targets cannot resolve a nonzero <=5% false-alarm rate, so
    require zero empirical false alarms. Never use the attack trace here.
    """
    clients = sorted(benign_residuals[0])
    peaks = {}
    for client in clients:
        z = [robust_z(round_values[client], [value for other, value in round_values.items() if other != client]) for round_values in benign_residuals]
        values, _ = cusum(z, kappa)
        peaks[client] = max(values, default=0.0)
    threshold = max(peaks.values()) + epsilon
    return threshold, peaks
