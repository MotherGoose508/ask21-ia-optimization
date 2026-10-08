"""Calculate the Section 6 optimum speeds for each weather profile.

This focused script uses the mathematical model already developed in the IA:

* Section 3 supplies ``a``, ``m`` and ``k`` for the lower-speed branch.
* Section 4 supplies ``d``, the transition speed and the higher-speed branch.
* ``data/ask21_polar.csv`` supplies the supported airspeed domain.
* ``data/profiles.csv`` supplies seven profiles interpolated through the
  Section 6 still-air, moderate and strong anchors.

For each branch, the script solves the numerator of R'(v) = 0 and compares
every valid stationary point with the branch endpoints.  A separate bounded
numerical optimization verifies each selected speed.  It exports the exact
six-column comparison table requested for Section 6 and a graph of R(v).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


PROJECT_ROOT = Path(__file__).resolve().parent
POLAR_PATH = PROJECT_ROOT / "data" / "ask21_polar.csv"
PROFILES_PATH = PROJECT_ROOT / "data" / "profiles.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "section_6"
TABLE_PATH = OUTPUT_DIR / "S6_branch_optimum_results.csv"
GRAPH_PATH = OUTPUT_DIR / "S6_profile_trends.png"
VERIFICATION_PATH = OUTPUT_DIR / "S6_branch_verification.txt"
JUSTIFICATION_PATH = OUTPUT_DIR / "S6_profile_spacing_justification.txt"

# Current IA model values (Sections 3 and 4).
A = 0.003822782
M = 21.5
K = 0.700
D_CORRECTION = 0.00330121
V_TRANSITION = 34.72  # 125 km/h, as stated in Section 4
VERIFICATION_TOLERANCE_MPS = 1e-5


@dataclass(frozen=True)
class QuadraticBranch:
    """A sink-rate branch written as s(v) = q*v^2 + b*v + c."""

    name: str
    q: float
    b: float
    c: float
    lower_bound: float
    upper_bound: float

    def sink(self, speed_mps: float | np.ndarray) -> float | np.ndarray:
        return self.q * speed_mps**2 + self.b * speed_mps + self.c


@dataclass(frozen=True)
class BranchOptimum:
    speed_mps: float
    range_ratio: float
    numerical_speed_mps: float


def make_branches(v_min: float, v_max: float) -> tuple[QuadraticBranch, QuadraticBranch]:
    """Expand the two IA sink-rate functions into quadratic coefficients."""

    lower = QuadraticBranch(
        name="lower",
        q=A,
        b=-2.0 * A * M,
        c=A * M**2 + K,
        lower_bound=v_min,
        upper_bound=V_TRANSITION,
    )
    higher = QuadraticBranch(
        name="higher",
        q=A - D_CORRECTION,
        b=-2.0 * A * M + 2.0 * D_CORRECTION * V_TRANSITION,
        c=A * M**2 + K - D_CORRECTION * V_TRANSITION**2,
        lower_bound=V_TRANSITION,
        upper_bound=v_max,
    )
    return lower, higher


def range_ratio(
    branch: QuadraticBranch,
    speed_mps: float | np.ndarray,
    headwind_mps: float,
    downdraft_mps: float,
) -> float | np.ndarray:
    """Return horizontal metres travelled per metre of altitude lost."""

    groundspeed = speed_mps - headwind_mps
    effective_sink = branch.sink(speed_mps) + downdraft_mps
    if np.any(np.asarray(groundspeed) <= 0):
        raise ValueError("The model requires v - H > 0.")
    if np.any(np.asarray(effective_sink) <= 0):
        raise ValueError("The model requires s(v) + D > 0.")
    return groundspeed / effective_sink


def stationary_speeds(
    branch: QuadraticBranch,
    headwind_mps: float,
    downdraft_mps: float,
) -> list[float]:
    """Solve the quadratic numerator of the derivative of R(v).

    If s(v) = q*v^2 + b*v + c, then

        numerator(R'(v)) = -q*v^2 + 2*q*H*v + c + D + b*H.
    """

    coefficients = [
        -branch.q,
        2.0 * branch.q * headwind_mps,
        branch.c + downdraft_mps + branch.b * headwind_mps,
    ]
    roots = np.roots(coefficients)
    return [float(root.real) for root in roots if abs(root.imag) < 1e-10]


def optimize_branch(
    branch: QuadraticBranch,
    headwind_mps: float,
    downdraft_mps: float,
) -> BranchOptimum:
    """Find the branch maximum from stationary points and endpoints."""

    candidates = [branch.lower_bound, branch.upper_bound]
    candidates.extend(stationary_speeds(branch, headwind_mps, downdraft_mps))
    candidates = [
        speed
        for speed in candidates
        if branch.lower_bound <= speed <= branch.upper_bound
        and speed > headwind_mps
        and branch.sink(speed) + downdraft_mps > 0
    ]
    if not candidates:
        raise ValueError(f"No valid speeds exist on the {branch.name} branch.")

    evaluated_candidates = [
        (
            speed,
            float(range_ratio(branch, speed, headwind_mps, downdraft_mps)),
        )
        for speed in candidates
    ]
    speed, maximum_range = max(evaluated_candidates, key=lambda item: item[1])

    # Independent numerical check over the same bounded branch.
    numerical = minimize_scalar(
        lambda value: -float(
            range_ratio(branch, value, headwind_mps, downdraft_mps)
        ),
        bounds=(branch.lower_bound, branch.upper_bound),
        method="bounded",
        options={"xatol": 1e-12},
    )
    numerical_candidates = [
        (branch.lower_bound, float(range_ratio(branch, branch.lower_bound, headwind_mps, downdraft_mps))),
        (float(numerical.x), -float(numerical.fun)),
        (branch.upper_bound, float(range_ratio(branch, branch.upper_bound, headwind_mps, downdraft_mps))),
    ]
    numerical_speed, _ = max(numerical_candidates, key=lambda item: item[1])
    if abs(speed - numerical_speed) > VERIFICATION_TOLERANCE_MPS:
        raise RuntimeError(
            f"Analytical and numerical optima disagree on the {branch.name} branch."
        )

    return BranchOptimum(speed, maximum_range, numerical_speed)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    polar = pd.read_csv(POLAR_PATH)
    profiles = pd.read_csv(PROFILES_PATH)
    required = {
        "profile_name",
        "severity_fraction",
        "headwind_mps",
        "downdraft_mps",
    }
    missing = required.difference(profiles.columns)
    if missing:
        raise ValueError(f"Missing profile columns: {sorted(missing)}")
    profiles = profiles.sort_values("severity_fraction", kind="stable").reset_index(drop=True)
    if not profiles["severity_fraction"].between(0.0, 1.0).all():
        raise ValueError("Profile severity fractions must lie between 0 and 1.")
    if not profiles["severity_fraction"].is_monotonic_increasing:
        raise ValueError("Profiles must be ordered from still air to strong.")
    if profiles["severity_fraction"].duplicated().any():
        raise ValueError("Profile severity fractions must be unique.")
    if not profiles["headwind_mps"].is_monotonic_increasing:
        raise ValueError("Headwind must not decrease as profile severity increases.")
    if not profiles["downdraft_mps"].is_monotonic_increasing:
        raise ValueError("Downdraft must not decrease as profile severity increases.")
    v_min = float(polar["airspeed_kmh"].min() / 3.6)
    v_max = float(polar["airspeed_kmh"].max() / 3.6)
    if not v_min < V_TRANSITION < v_max:
        raise ValueError("The transition speed must lie inside the polar domain.")
    return polar, profiles, v_min, v_max


def calculate_results() -> tuple[pd.DataFrame, list[dict], tuple[QuadraticBranch, QuadraticBranch]]:
    _, profiles, v_min, v_max = load_inputs()
    lower, higher = make_branches(v_min, v_max)
    rows: list[dict] = []
    detailed: list[dict] = []

    for profile in profiles.itertuples(index=False):
        low = optimize_branch(lower, profile.headwind_mps, profile.downdraft_mps)
        high = optimize_branch(higher, profile.headwind_mps, profile.downdraft_mps)
        best = max((low, high), key=lambda result: result.range_ratio)
        rows.append(
            {
                "profile": profile.profile_name,
                "headwind_mps": float(profile.headwind_mps),
                "downdraft_mps": float(profile.downdraft_mps),
                "lower_branch_optimum_mps": low.speed_mps,
                "higher_branch_optimum_mps": high.speed_mps,
                "maximum_range_m_per_m_altitude": best.range_ratio,
            }
        )
        detailed.append(
            {
                "profile": profile.profile_name,
                "severity_fraction": float(profile.severity_fraction),
                "headwind_mps": float(profile.headwind_mps),
                "downdraft_mps": float(profile.downdraft_mps),
                "lower": low,
                "higher": high,
            }
        )

    return pd.DataFrame(rows), detailed, (lower, higher)


def plot_results(
    detailed: list[dict],
    branches: tuple[QuadraticBranch, QuadraticBranch],
) -> None:
    _ = branches  # The transition value is already fixed by the current IA.
    positions = np.arange(len(detailed))
    labels = [record["profile"].replace(" Profile", "") for record in detailed]
    lower_speeds = np.array([record["lower"].speed_mps for record in detailed])
    higher_speeds = np.array([record["higher"].speed_mps for record in detailed])
    lower_ranges = np.array([record["lower"].range_ratio for record in detailed])
    higher_ranges = np.array([record["higher"].range_ratio for record in detailed])
    maximum_ranges = np.maximum(lower_ranges, higher_ranges)
    selected_speeds = np.where(
        lower_ranges >= higher_ranges, lower_speeds, higher_speeds
    )

    fig, (speed_ax, range_ax) = plt.subplots(1, 2, figsize=(12.2, 5.4))
    speed_ax.plot(
        positions, lower_speeds, color="#0072B2", linewidth=2.2,
        marker="o", markersize=6.5, label="Lower-branch optimum"
    )
    speed_ax.plot(
        positions, higher_speeds, color="#E69F00", linewidth=2.2,
        marker="s", markersize=6.5, linestyle="--", label="Higher-branch optimum"
    )
    speed_ax.scatter(
        positions, selected_speeds, facecolors="white", edgecolors="#222222",
        linewidths=1.4, marker="*", s=130, zorder=5, label="Global optimum"
    )
    speed_ax.axhline(
        V_TRANSITION, color="#666666", linewidth=1.2, linestyle=":",
        label=f"Transition ({V_TRANSITION:.2f} m/s)"
    )
    speed_ax.set_title("Optimum airspeed by branch")
    speed_ax.set_ylabel("Optimum airspeed (m/s)")
    speed_ax.legend(frameon=False, fontsize=9)

    range_ax.plot(
        positions, maximum_ranges, color="#6A4C93", linewidth=2.4,
        marker="o", markersize=7
    )
    for position, value in zip(positions, maximum_ranges):
        range_ax.annotate(
            f"{value:.2f}", (position, value), xytext=(0, 8),
            textcoords="offset points", ha="center", fontsize=8.5
        )
    range_ax.set_title("Maximum range ratio")
    range_ax.set_ylabel("Range (m per m of altitude lost)")

    for axis in (speed_ax, range_ax):
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.set_xlabel("Profile severity from still air to strong")
        axis.grid(True, axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "ASK 21 Section 6 profile sweep",
        fontsize=16,
        y=1.02,
    )
    fig.text(
        0.5, -0.03,
        "Seven levels at one-sixth severity intervals; still, moderate and strong are retained as anchors.",
        ha="center", fontsize=9, color="#555555"
    )
    fig.tight_layout()
    fig.savefig(GRAPH_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results, detailed, branches = calculate_results()
    results.round(6).to_csv(TABLE_PATH, index=False, lineterminator="\n")
    plot_results(detailed, branches)

    verification_lines = [
        "Section 6 analytical versus bounded numerical verification",
        f"Tolerance: {VERIFICATION_TOLERANCE_MPS:g} m/s",
    ]
    for record in detailed:
        low_difference = abs(
            record["lower"].speed_mps - record["lower"].numerical_speed_mps
        )
        high_difference = abs(
            record["higher"].speed_mps - record["higher"].numerical_speed_mps
        )
        verification_lines.append(
            f"{record['profile']}: lower difference={low_difference:.12g} m/s; "
            f"higher difference={high_difference:.12g} m/s; PASS"
        )
    VERIFICATION_PATH.write_text("\n".join(verification_lines) + "\n", encoding="utf-8")

    JUSTIFICATION_PATH.write_text(
        "\n".join(
            [
                "Section 6 profile spacing justification",
                "",
                "Anchors retained from the current IA:",
                "- Still air: H = 0 m/s, D = 0 m/s",
                "- Moderate: H = 7 m/s, D = 0.5 m/s",
                "- Strong: H = 12 m/s, D = 1 m/s",
                "",
                "The sweep uses seven severity levels t = 0, 1/6, ..., 1.",
                "Downdraft increases uniformly by 1/6 m/s per level.",
                "From still to moderate, headwind increases by 7/3 m/s per level.",
                "From moderate to strong, headwind increases by 5/3 m/s per level.",
                "This piecewise spacing preserves the sourced moderate anchor at 7 m/s;",
                "a single 2 m/s grid would replace it with an invented 6 m/s value.",
                "Seven profiles are dense enough to reveal curvature and the branch switch",
                "without making the table and graph unnecessarily crowded.",
                "",
                "These intermediate pairs are controlled interpolated scenarios, not",
                "claims that headwind and downdraft always occur together in nature.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(results.round(3).to_string(index=False))
    print(f"\nSaved table: {TABLE_PATH}")
    print(f"Saved graph: {GRAPH_PATH}")


if __name__ == "__main__":
    main()
