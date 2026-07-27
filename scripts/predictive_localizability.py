#!/usr/bin/env python3
"""Direction-aware observability utilities for Skyline Lock and TRN.

The existing demos use a scalar uniqueness margin: it answers whether another
place matches nearly as well.  This module answers a different, complementary
question before the factor-graph solve: which state directions does the raw
measurement constrain locally?

All information matrices are reported in scaled coordinates.  For Skyline the
default perturbation coordinates are ``[x / position_scale_m,
y / position_scale_m, yaw / yaw_scale_rad]``.  This makes translation and yaw
comparable without pretending that metres and radians have the same units.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ObservabilityReport:
    """Serializable local observability result."""

    state_labels: tuple[str, ...]
    information_matrix: Array
    scaled_information_matrix: Array
    eigenvalues: Array
    weak_direction: Array
    directional_confidence: Array
    spectral_balance: float
    measurement_strength: float
    alignment_risk: float
    state_scale: Array

    def to_dict(self) -> dict:
        result = asdict(self)
        for key, value in tuple(result.items()):
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, tuple):
                result[key] = list(value)
        return result


def _camera_frame_profile(world_profile: Array, yaw_rad: float) -> Array:
    """Continuously rotate a world-frame circular profile into the camera frame."""
    profile = np.asarray(world_profile, dtype=np.float64)
    n = profile.size
    sample = (np.arange(n, dtype=np.float64) + yaw_rad * n / (2.0 * np.pi)) % n
    return np.interp(sample, np.arange(n, dtype=np.float64), profile, period=n)


def skyline_measurement_jacobian(
    render_world_horizon: Callable[[tuple[float, float]], Array],
    xy_m: Sequence[float],
    yaw_rad: float,
    *,
    position_delta_m: float,
    yaw_delta_rad: float,
) -> tuple[Array, Array]:
    """Return ``(reference_profile, J)`` for state ``(x, y, yaw)``.

    Central differences operate directly on predicted raw horizon profiles, so
    no optimizer result or posterior covariance is needed.
    """
    if position_delta_m <= 0.0 or yaw_delta_rad <= 0.0:
        raise ValueError("finite-difference deltas must be positive")

    x, y = float(xy_m[0]), float(xy_m[1])

    def observe(px: float, py: float, heading: float) -> Array:
        return _camera_frame_profile(render_world_horizon((px, py)), heading)

    reference = observe(x, y, yaw_rad)
    dx = (
        observe(x + position_delta_m, y, yaw_rad)
        - observe(x - position_delta_m, y, yaw_rad)
    ) / (2.0 * position_delta_m)
    dy = (
        observe(x, y + position_delta_m, yaw_rad)
        - observe(x, y - position_delta_m, yaw_rad)
    ) / (2.0 * position_delta_m)
    dyaw = (
        observe(x, y, yaw_rad + yaw_delta_rad)
        - observe(x, y, yaw_rad - yaw_delta_rad)
    ) / (2.0 * yaw_delta_rad)
    return reference, np.column_stack((dx, dy, dyaw))


def report_from_jacobian(
    jacobian: Array,
    *,
    measurement_sigma: float,
    state_scale: Sequence[float],
    state_labels: Sequence[str],
) -> ObservabilityReport:
    """Convert a measurement Jacobian into a direction-aware risk report."""
    J = np.asarray(jacobian, dtype=np.float64)
    scale = np.asarray(state_scale, dtype=np.float64)
    labels = tuple(state_labels)
    if J.ndim != 2 or J.shape[1] != scale.size or scale.size != len(labels):
        raise ValueError("jacobian columns, state_scale, and state_labels must agree")
    if measurement_sigma <= 0.0 or np.any(scale <= 0.0):
        raise ValueError("measurement_sigma and state scales must be positive")

    information = (J.T @ J) / (measurement_sigma * measurement_sigma)
    D = np.diag(scale)
    scaled = D @ information @ D
    scaled = 0.5 * (scaled + scaled.T)
    eigenvalues, eigenvectors = np.linalg.eigh(scaled)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    largest = max(float(eigenvalues[-1]), 1e-15)
    balance = float(eigenvalues[0] / largest)
    # One scaled-state unit in the weakest direction. This separates "balanced
    # but no signal" (flat horizon) from genuinely informative measurements.
    strength = float(1.0 - np.exp(-eigenvalues[0]))
    reliability = float(np.sqrt(np.clip(balance * strength, 0.0, 1.0)))
    diag = np.maximum(np.diag(scaled), 0.0)
    directional = diag / max(float(diag.max()), 1e-15)

    return ObservabilityReport(
        state_labels=labels,
        information_matrix=information,
        scaled_information_matrix=scaled,
        eigenvalues=eigenvalues,
        weak_direction=eigenvectors[:, 0],
        directional_confidence=directional,
        spectral_balance=balance,
        measurement_strength=strength,
        alignment_risk=1.0 - reliability,
        state_scale=scale,
    )


def skyline_observability(
    render_world_horizon: Callable[[tuple[float, float]], Array],
    xy_m: Sequence[float],
    yaw_rad: float,
    *,
    position_delta_m: float,
    yaw_delta_rad: float,
    horizon_sigma_rad: float,
    position_scale_m: float | None = None,
    yaw_scale_rad: float | None = None,
) -> tuple[Array, ObservabilityReport]:
    """Predict local ``(x, y, yaw)`` observability from a Skyline measurement."""
    reference, J = skyline_measurement_jacobian(
        render_world_horizon,
        xy_m,
        yaw_rad,
        position_delta_m=position_delta_m,
        yaw_delta_rad=yaw_delta_rad,
    )
    report = report_from_jacobian(
        J,
        measurement_sigma=horizon_sigma_rad,
        state_scale=(
            position_scale_m or position_delta_m,
            position_scale_m or position_delta_m,
            yaw_scale_rad or yaw_delta_rad,
        ),
        state_labels=("x", "y", "yaw"),
    )
    return reference, report


def skyline_grid_observability(
    predicted_profiles: Array,
    *,
    grid_shape: tuple[int, int],
    grid_step_m: float,
    horizon_sigma_rad: float,
    position_scale_m: float | None = None,
    yaw_scale_rad: float,
) -> tuple[Array, list[ObservabilityReport]]:
    """Compute a directional reliability map from an existing horizon grid.

    Spatial derivatives reuse neighboring predicted profiles, so a routing map
    needs no additional horizon renders beyond the existing uniqueness map.
    """
    rows, cols = grid_shape
    profiles = np.asarray(predicted_profiles, dtype=np.float64)
    if profiles.ndim != 2 or profiles.shape[0] != rows * cols:
        raise ValueError("predicted_profiles and grid_shape do not agree")
    if grid_step_m <= 0.0 or yaw_scale_rad <= 0.0:
        raise ValueError("grid and yaw scales must be positive")
    cube = profiles.reshape(rows, cols, profiles.shape[1])
    edge_order = 2 if min(rows, cols) >= 3 else 1
    dy, dx = np.gradient(cube, grid_step_m, grid_step_m,
                         axis=(0, 1), edge_order=edge_order)
    azimuth_step = 2.0 * np.pi / profiles.shape[1]
    dyaw = (
        np.roll(cube, -1, axis=2) - np.roll(cube, 1, axis=2)
    ) / (2.0 * azimuth_step)

    reports = []
    reliability = np.zeros((rows, cols), dtype=np.float64)
    for row in range(rows):
        for col in range(cols):
            report = report_from_jacobian(
                np.column_stack((dx[row, col], dy[row, col], dyaw[row, col])),
                measurement_sigma=horizon_sigma_rad,
                state_scale=(
                    position_scale_m or grid_step_m,
                    position_scale_m or grid_step_m,
                    yaw_scale_rad,
                ),
                state_labels=("x", "y", "yaw"),
            )
            reports.append(report)
            reliability[row, col] = 1.0 - report.alignment_risk
    return reliability, reports


def trn_response_jacobian(
    response: Array,
    peak_rc: Sequence[int],
    *,
    px_to_m: float,
) -> Array:
    """Return a square-root curvature matrix around a TRN response peak.

    The negative Hessian of the score surface is positive at a well-defined
    maximum and acts as local match information. Its PSD square root is returned
    as a Jacobian-like matrix for the common reporting path. A peak on the
    response boundary is deliberately reported as unconstrained.
    """
    score = np.asarray(response, dtype=np.float64)
    row, col = int(peak_rc[0]), int(peak_rc[1])
    if px_to_m <= 0.0:
        raise ValueError("px_to_m must be positive")
    if row < 1 or col < 1 or row >= score.shape[0] - 1 or col >= score.shape[1] - 1:
        return np.zeros((2, 2), dtype=np.float64)
    centre = score[row, col]
    inv_h2 = 1.0 / (px_to_m * px_to_m)
    hxx = -(score[row, col + 1] - 2.0 * centre + score[row, col - 1]) * inv_h2
    hyy = -(score[row + 1, col] - 2.0 * centre + score[row - 1, col]) * inv_h2
    hxy = -(
        score[row + 1, col + 1] - score[row + 1, col - 1]
        - score[row - 1, col + 1] + score[row - 1, col - 1]
    ) / (4.0 * px_to_m * px_to_m)
    curvature = np.array([[hxx, hxy], [hxy, hyy]], dtype=np.float64)
    values, vectors = np.linalg.eigh(0.5 * (curvature + curvature.T))
    values = np.maximum(values, 0.0)
    return np.diag(np.sqrt(values)) @ vectors.T


def trn_observability(
    response: Array,
    peak_rc: Sequence[int],
    *,
    px_to_m: float,
    score_sigma: float = 0.01,
    position_scale_m: float | None = None,
) -> ObservabilityReport:
    """Predict local XY observability from a raw TRN match response."""
    J = trn_response_jacobian(response, peak_rc, px_to_m=px_to_m)
    scale = position_scale_m or px_to_m
    return report_from_jacobian(
        J,
        measurement_sigma=score_sigma,
        state_scale=(scale, scale),
        state_labels=("x", "y"),
    )


def normalized_information(
    report: ObservabilityReport,
    labels: Sequence[str],
    *,
    sigma_best: float,
    reliability: float = 1.0,
    information_floor: float = 0.0,
) -> Array:
    """Extract a directional factor information matrix with controlled scale.

    The raw Fisher matrix supplies directions, while ``sigma_best`` supplies the
    mission-facing metric scale. This prevents finite-difference or sensor-noise
    units from silently changing the absolute factor weight.
    """
    if sigma_best <= 0.0:
        raise ValueError("sigma_best must be positive")
    indices = [report.state_labels.index(label) for label in labels]
    block = report.scaled_information_matrix[np.ix_(indices, indices)]
    values, vectors = np.linalg.eigh(0.5 * (block + block.T))
    largest = max(float(values[-1]), 1e-15)
    normalized_values = np.clip(values / largest, information_floor, 1.0)
    bounded_reliability = float(np.clip(reliability, 0.0, 1.0))
    return (
        bounded_reliability
        * (vectors @ np.diag(normalized_values) @ vectors.T)
        / (sigma_best * sigma_best)
    )


def fuse_positions_directional(
    n_poses: int,
    vo_deltas: Array,
    prior_xy: Sequence[float],
    *,
    sigma_prior: float,
    sigma_vo: float,
    fixes: Sequence[tuple[int, Array, Array]],
) -> tuple[Array, Array]:
    """Linear XY pose graph with full 2x2 information for every absolute fix."""
    if n_poses < 1 or sigma_prior <= 0.0 or sigma_vo <= 0.0:
        raise ValueError("invalid graph dimensions or noise")
    H = np.zeros((2 * n_poses, 2 * n_poses), dtype=np.float64)
    b = np.zeros(2 * n_poses, dtype=np.float64)

    def block(k: int) -> slice:
        return slice(2 * k, 2 * k + 2)

    prior_info = np.eye(2) / (sigma_prior * sigma_prior)
    H[block(0), block(0)] += prior_info
    b[block(0)] += prior_info @ np.asarray(prior_xy, dtype=np.float64)

    between_info = np.eye(2) / (sigma_vo * sigma_vo)
    deltas = np.asarray(vo_deltas, dtype=np.float64)
    if deltas.shape != (n_poses - 1, 2):
        raise ValueError("vo_deltas must have shape (n_poses - 1, 2)")
    for k, delta in enumerate(deltas, start=1):
        prev, cur = block(k - 1), block(k)
        H[prev, prev] += between_info
        H[cur, cur] += between_info
        H[prev, cur] -= between_info
        H[cur, prev] -= between_info
        b[prev] -= between_info @ delta
        b[cur] += between_info @ delta

    for k, measurement, information in fixes:
        if k < 0 or k >= n_poses:
            raise ValueError(f"fix pose index out of range: {k}")
        info = np.asarray(information, dtype=np.float64)
        if info.shape != (2, 2):
            raise ValueError("fix information must be 2x2")
        info = 0.5 * (info + info.T)
        H[block(k), block(k)] += info
        b[block(k)] += info @ np.asarray(measurement, dtype=np.float64)

    covariance = np.linalg.inv(H)
    estimate = np.linalg.solve(H, b).reshape(n_poses, 2)
    pose_covariance = np.stack([
        covariance[block(k), block(k)] for k in range(n_poses)
    ])
    return estimate, pose_covariance
