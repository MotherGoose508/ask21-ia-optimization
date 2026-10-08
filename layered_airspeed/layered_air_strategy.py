"""Optimize an ASK 21 airspeed separately in each altitude layer.

The program implements the piecewise sink-rate model in the current IA:

    s1(v) = a(v - m)^2 + k
    s2(v) = a(v - m)^2 + k - d(v - v_transition)^2

For a layer with headwind H, downdraft D and altitude loss delta_h,

    R(v) = (v - H) / (s(v) + D)
    x     = delta_h * R(v)

Each layer is optimized independently.  The resulting total distance is then
compared with the best strategy that must use one constant airspeed in every
layer.  CSV outputs keep the branch, speed and distance calculations visible.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "model_parameters.csv"
DEFAULT_LAYERS_PATH = SCRIPT_DIR / "layers.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"


@dataclass(frozen=True)
class ModelParameters:
    a: float
    m_mps: float
    k_mps: float
    d: float
    transition_speed_mps: float
    minimum_speed_mps: float
    maximum_speed_mps: float

    def validate(self) -> None:
        if self.a <= 0:
            raise ValueError("Model parameter a must be positive.")
        if self.d < 0:
            raise ValueError("Model parameter d cannot be negative.")
        if self.a - self.d <= 0:
            raise ValueError("The higher-speed branch requires a - d > 0.")
        if not self.minimum_speed_mps < self.transition_speed_mps < self.maximum_speed_mps:
            raise ValueError("The transition speed must lie strictly inside the speed domain.")
        if self.k_mps <= 0:
            raise ValueError("The minimum sink-rate parameter k must be positive.")


@dataclass(frozen=True)
class Layer:
    name: str
    altitude_loss_m: float
    headwind_mps: float
    downdraft_mps: float
    source_status: str = ""


@dataclass(frozen=True)
class Branch:
    code: str
    label: str
    q: float
    b: float
    c: float
    lower_speed_mps: float
    upper_speed_mps: float

    def sink_rate(self, speed_mps: float) -> float:
        return self.q * speed_mps**2 + self.b * speed_mps + self.c

    def derivative(self, speed_mps: float) -> float:
        return 2.0 * self.q * speed_mps + self.b


@dataclass(frozen=True)
class Optimum:
    speed_mps: float
    range_ratio: float
    branch_code: str
    branch_label: str
    source: str
    numerical_speed_mps: float


def make_branches(model: ModelParameters) -> tuple[Branch, Branch]:
    """Expand both IA sink-rate branches into q*v^2 + b*v + c."""

    lower = Branch(
        code="s1",
        label="lower-speed branch",
        q=model.a,
        b=-2.0 * model.a * model.m_mps,
        c=model.a * model.m_mps**2 + model.k_mps,
        lower_speed_mps=model.minimum_speed_mps,
        upper_speed_mps=model.transition_speed_mps,
    )
    higher = Branch(
        code="s2",
        label="higher-speed branch",
        q=model.a - model.d,
        b=-2.0 * model.a * model.m_mps
        + 2.0 * model.d * model.transition_speed_mps,
        c=model.a * model.m_mps**2
        + model.k_mps
        - model.d * model.transition_speed_mps**2,
        lower_speed_mps=model.transition_speed_mps,
        upper_speed_mps=model.maximum_speed_mps,
    )
    return lower, higher


def load_model(path: Path) -> ModelParameters:
    required = {
        "a",
        "m_mps",
        "k_mps",
        "d",
        "transition_speed_mps",
        "minimum_speed_mps",
        "maximum_speed_mps",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one model row.")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing model columns: {sorted(missing)}")
    try:
        model = ModelParameters(**{name: float(rows[0][name]) for name in required})
    except (TypeError, ValueError) as error:
        raise ValueError(f"All model parameters in {path} must be numeric.") from error
    model.validate()
    return model


def load_layers(path: Path) -> list[Layer]:
    required = {"layer", "altitude_loss_m", "headwind_mps", "downdraft_mps"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing layer columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} must contain at least one altitude layer.")

    layers: list[Layer] = []
    seen_names: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        name = (row.get("layer") or "").strip()
        if not name:
            raise ValueError(f"Layer name is blank on CSV line {line_number}.")
        if name in seen_names:
            raise ValueError(f"Layer name '{name}' is duplicated.")
        seen_names.add(name)
        try:
            altitude = float(row["altitude_loss_m"])
            headwind = float(row["headwind_mps"])
            downdraft = float(row["downdraft_mps"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Layer values must be numeric on CSV line {line_number}.") from error
        if altitude <= 0:
            raise ValueError(f"Altitude loss must be positive for layer '{name}'.")
        if headwind < 0 or downdraft < 0:
            raise ValueError(f"Headwind and downdraft cannot be negative for layer '{name}'.")
        layers.append(
            Layer(
                name=name,
                altitude_loss_m=altitude,
                headwind_mps=headwind,
                downdraft_mps=downdraft,
                source_status=(row.get("source_status") or "").strip(),
            )
        )
    return layers


def range_ratio(branch: Branch, speed_mps: float, headwind_mps: float, downdraft_mps: float) -> float:
    """Return horizontal metres per metre of altitude lost."""

    ground_speed = speed_mps - headwind_mps
    effective_sink = branch.sink_rate(speed_mps) + downdraft_mps
    if ground_speed <= 0:
        raise ValueError("The range model requires v - H > 0.")
    if effective_sink <= 0:
        raise ValueError("The range model requires s(v) + D > 0.")
    return ground_speed / effective_sink


def stationary_speeds(branch: Branch, headwind_mps: float, downdraft_mps: float) -> list[float]:
    """Solve the numerator of R'(v)=0 for one quadratic sink-rate branch.

    If s(v)=q*v^2+b*v+c, the derivative numerator simplifies to

        -q*v^2 + 2*q*H*v + c + D + b*H = 0.
    """

    a2 = -branch.q
    a1 = 2.0 * branch.q * headwind_mps
    a0 = branch.c + downdraft_mps + branch.b * headwind_mps
    discriminant = a1**2 - 4.0 * a2 * a0
    if discriminant < -1e-12:
        return []
    discriminant = max(0.0, discriminant)
    root = math.sqrt(discriminant)
    return sorted(((-a1 - root) / (2.0 * a2), (-a1 + root) / (2.0 * a2)))


def _feasible(branch: Branch, speed_mps: float, layer: Layer) -> bool:
    return (
        branch.lower_speed_mps - 1e-12 <= speed_mps <= branch.upper_speed_mps + 1e-12
        and speed_mps > layer.headwind_mps
        and branch.sink_rate(speed_mps) + layer.downdraft_mps > 0
    )


def golden_section_maximize(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    tolerance: float = 1e-12,
    maximum_iterations: int = 300,
) -> tuple[float, float]:
    """Maximize a continuous function on one short bounded interval."""

    if upper < lower:
        raise ValueError("Upper optimization bound is below the lower bound.")
    if math.isclose(lower, upper, abs_tol=tolerance, rel_tol=0.0):
        return lower, objective(lower)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = upper - ratio * (upper - lower)
    x2 = lower + ratio * (upper - lower)
    f1 = objective(x1)
    f2 = objective(x2)
    for _ in range(maximum_iterations):
        if upper - lower <= tolerance:
            break
        if f1 < f2:
            lower = x1
            x1, f1 = x2, f2
            x2 = lower + ratio * (upper - lower)
            f2 = objective(x2)
        else:
            upper = x2
            x2, f2 = x1, f1
            x1 = upper - ratio * (upper - lower)
            f1 = objective(x1)
    candidates = [(lower, objective(lower)), (x1, f1), (x2, f2), (upper, objective(upper))]
    return max(candidates, key=lambda item: item[1])


def optimize_branch(branch: Branch, layer: Layer) -> Optimum:
    """Compare derivative roots and endpoints, then verify numerically."""

    candidates: list[tuple[float, str]] = [
        (branch.lower_speed_mps, "branch endpoint"),
        (branch.upper_speed_mps, "branch endpoint"),
    ]
    candidates.extend(
        (speed, "stationary point from R'(v)=0")
        for speed in stationary_speeds(branch, layer.headwind_mps, layer.downdraft_mps)
    )
    evaluated = [
        (
            speed,
            range_ratio(branch, speed, layer.headwind_mps, layer.downdraft_mps),
            source,
        )
        for speed, source in candidates
        if _feasible(branch, speed, layer)
    ]
    if not evaluated:
        raise ValueError(f"No physically valid speed exists on {branch.label} for layer '{layer.name}'.")
    analytical_speed, analytical_ratio, source = max(evaluated, key=lambda item: item[1])

    numerical_lower = max(branch.lower_speed_mps, math.nextafter(layer.headwind_mps, math.inf))
    if numerical_lower > branch.upper_speed_mps:
        numerical_speed = analytical_speed
    else:
        interior_speed, interior_ratio = golden_section_maximize(
            lambda speed: range_ratio(branch, speed, layer.headwind_mps, layer.downdraft_mps),
            numerical_lower,
            branch.upper_speed_mps,
        )
        numerical_candidates = [
            (
                numerical_lower,
                range_ratio(
                    branch,
                    numerical_lower,
                    layer.headwind_mps,
                    layer.downdraft_mps,
                ),
            ),
            (interior_speed, interior_ratio),
            (
                branch.upper_speed_mps,
                range_ratio(
                    branch,
                    branch.upper_speed_mps,
                    layer.headwind_mps,
                    layer.downdraft_mps,
                ),
            ),
        ]
        numerical_speed, _ = max(numerical_candidates, key=lambda item: item[1])

    if abs(analytical_speed - numerical_speed) > 1e-5:
        raise RuntimeError(
            f"Analytical and numerical optima disagree on {branch.label} in layer '{layer.name}'."
        )
    return Optimum(
        speed_mps=analytical_speed,
        range_ratio=analytical_ratio,
        branch_code=branch.code,
        branch_label=branch.label,
        source=source,
        numerical_speed_mps=numerical_speed,
    )


def optimize_layer(branches: tuple[Branch, Branch], layer: Layer) -> tuple[Optimum, list[Optimum]]:
    branch_optima: list[Optimum] = []
    for branch in branches:
        try:
            branch_optima.append(optimize_branch(branch, layer))
        except ValueError:
            continue
    if not branch_optima:
        raise ValueError(f"No valid airspeed exists for layer '{layer.name}'.")
    best = max(branch_optima, key=lambda result: result.range_ratio)
    return best, branch_optima


def branch_for_speed(branches: tuple[Branch, Branch], speed_mps: float) -> Branch:
    return branches[0] if speed_mps <= branches[0].upper_speed_mps else branches[1]


def total_distance_at_speed(branches: tuple[Branch, Branch], layers: Iterable[Layer], speed_mps: float) -> float:
    branch = branch_for_speed(branches, speed_mps)
    total = 0.0
    for layer in layers:
        total += layer.altitude_loss_m * range_ratio(
            branch, speed_mps, layer.headwind_mps, layer.downdraft_mps
        )
    return total


def _grid_refined_maximum(
    objective: Callable[[float], float], lower: float, upper: float, grid_steps: int = 20000
) -> tuple[float, float]:
    """Find all grid-local peaks and refine them, avoiding a unimodality assumption."""

    if upper <= lower:
        return lower, objective(lower)
    width = (upper - lower) / grid_steps
    speeds = [lower + index * width for index in range(grid_steps + 1)]
    values = [objective(speed) for speed in speeds]
    peak_indices = {0, grid_steps}
    for index in range(1, grid_steps):
        if values[index] >= values[index - 1] and values[index] >= values[index + 1]:
            peak_indices.add(index)
    candidates = [(speeds[index], values[index]) for index in peak_indices]
    for index in peak_indices:
        left = speeds[max(0, index - 1)]
        right = speeds[min(grid_steps, index + 1)]
        candidates.append(golden_section_maximize(objective, left, right))
    return max(candidates, key=lambda item: item[1])


def optimize_constant_speed(
    model: ModelParameters, branches: tuple[Branch, Branch], layers: list[Layer]
) -> tuple[float, float, str]:
    """Find the best one-speed strategy across every supplied layer."""

    feasible_lower = max(model.minimum_speed_mps, max(layer.headwind_mps for layer in layers))
    feasible_lower = math.nextafter(feasible_lower, math.inf)
    candidates: list[tuple[float, float]] = []
    for branch in branches:
        lower = max(branch.lower_speed_mps, feasible_lower)
        upper = branch.upper_speed_mps
        if lower > upper:
            continue
        candidates.append(
            _grid_refined_maximum(
                lambda speed: total_distance_at_speed(branches, layers, speed), lower, upper
            )
        )
    if not candidates:
        raise ValueError("No one-speed strategy has positive ground speed in every layer.")
    speed, distance = max(candidates, key=lambda item: item[1])
    selected_branch = branch_for_speed(branches, speed)
    return speed, distance, selected_branch.code


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def calculate(
    model: ModelParameters, layers: list[Layer]
) -> tuple[list[dict], list[dict], dict, list[str]]:
    branches = make_branches(model)
    layer_rows: list[dict] = []
    candidate_rows: list[dict] = []
    verification_lines = [
        "Layer-specific analytical versus numerical verification",
        "Tolerance: 1e-5 m/s",
    ]

    optimized_total = 0.0
    optimized_results: list[tuple[Layer, Optimum]] = []
    for layer_number, layer in enumerate(layers, start=1):
        optimum, branch_optima = optimize_layer(branches, layer)
        optimized_results.append((layer, optimum))
        optimized_total += layer.altitude_loss_m * optimum.range_ratio
        for branch_result in branch_optima:
            candidate_rows.append(
                {
                    "layer_number": layer_number,
                    "layer": layer.name,
                    "branch": branch_result.branch_code,
                    "branch_label": branch_result.branch_label,
                    "candidate_speed_mps": branch_result.speed_mps,
                    "candidate_speed_kmh": branch_result.speed_mps * 3.6,
                    "range_ratio_m_per_m_altitude": branch_result.range_ratio,
                    "candidate_source": branch_result.source,
                    "selected_for_layer": branch_result is optimum,
                }
            )
            difference = abs(branch_result.speed_mps - branch_result.numerical_speed_mps)
            verification_lines.append(
                f"{layer.name}, {branch_result.branch_code}: difference={difference:.12g} m/s; PASS"
            )

    constant_speed, constant_total, constant_branch = optimize_constant_speed(
        model, branches, layers
    )
    still_air_layer = Layer("still-air reference", 1.0, 0.0, 0.0)
    still_air_optimum, _ = optimize_layer(branches, still_air_layer)
    still_air_constant_total = total_distance_at_speed(
        branches, layers, still_air_optimum.speed_mps
    )

    for layer_number, (layer, optimum) in enumerate(optimized_results, start=1):
        branch = branch_for_speed(branches, optimum.speed_mps)
        sink_rate = branch.sink_rate(optimum.speed_mps)
        constant_layer_distance = layer.altitude_loss_m * range_ratio(
            branch_for_speed(branches, constant_speed),
            constant_speed,
            layer.headwind_mps,
            layer.downdraft_mps,
        )
        optimized_distance = layer.altitude_loss_m * optimum.range_ratio
        selected_branch = optimum.branch_code
        if math.isclose(
            optimum.speed_mps, model.transition_speed_mps, abs_tol=1e-9, rel_tol=0.0
        ):
            selected_branch = "transition (s1 = s2)"
        layer_rows.append(
            {
                "layer_number": layer_number,
                "layer": layer.name,
                "source_status": layer.source_status,
                "altitude_loss_m": layer.altitude_loss_m,
                "headwind_mps": layer.headwind_mps,
                "downdraft_mps": layer.downdraft_mps,
                "selected_branch": selected_branch,
                "optimal_airspeed_mps": optimum.speed_mps,
                "optimal_airspeed_kmh": optimum.speed_mps * 3.6,
                "ground_speed_mps": optimum.speed_mps - layer.headwind_mps,
                "sink_rate_mps": sink_rate,
                "effective_sink_rate_mps": sink_rate + layer.downdraft_mps,
                "range_ratio_m_per_m_altitude": optimum.range_ratio,
                "layer_specific_distance_m": optimized_distance,
                "constant_airspeed_mps": constant_speed,
                "constant_layer_distance_m": constant_layer_distance,
                "layer_specific_advantage_m": optimized_distance - constant_layer_distance,
            }
        )

    improvement = optimized_total - constant_total
    comparison = {
        "total_altitude_loss_m": sum(layer.altitude_loss_m for layer in layers),
        "layer_specific_total_distance_m": optimized_total,
        "optimized_constant_airspeed_mps": constant_speed,
        "optimized_constant_airspeed_kmh": constant_speed * 3.6,
        "optimized_constant_selected_branch": constant_branch,
        "optimized_constant_total_distance_m": constant_total,
        "layer_specific_improvement_m": improvement,
        "layer_specific_improvement_percent": 100.0 * improvement / constant_total,
        "still_air_reference_airspeed_mps": still_air_optimum.speed_mps,
        "still_air_reference_airspeed_kmh": still_air_optimum.speed_mps * 3.6,
        "still_air_reference_total_distance_m": still_air_constant_total,
        "improvement_vs_still_air_reference_m": optimized_total - still_air_constant_total,
        "improvement_vs_still_air_reference_percent": 100.0
        * (optimized_total - still_air_constant_total)
        / still_air_constant_total,
    }
    if optimized_total + 1e-7 < constant_total:
        raise RuntimeError("Layer-specific optimization cannot underperform its one-speed subset.")
    return layer_rows, candidate_rows, comparison, verification_lines


def run(model_path: Path, layers_path: Path, output_dir: Path) -> dict:
    model = load_model(model_path)
    layers = load_layers(layers_path)
    layer_rows, candidate_rows, comparison, verification_lines = calculate(model, layers)

    write_csv(output_dir / "layer_results.csv", list(layer_rows[0]), layer_rows)
    write_csv(output_dir / "layer_branch_candidates.csv", list(candidate_rows[0]), candidate_rows)
    write_csv(output_dir / "strategy_comparison.csv", list(comparison), [comparison])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "verification.txt").write_text(
        "\n".join(verification_lines) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.txt").write_text(
        "\n".join(
            [
                "ASK 21 layered airspeed strategy",
                "",
                f"Layers: {len(layers)}",
                f"Total altitude loss: {comparison['total_altitude_loss_m']:.3f} m",
                f"Layer-specific distance: {comparison['layer_specific_total_distance_m']:.3f} m",
                "Best constant airspeed: "
                f"{comparison['optimized_constant_airspeed_mps']:.6f} m/s "
                f"({comparison['optimized_constant_airspeed_kmh']:.3f} km/h)",
                f"Best constant-speed distance: {comparison['optimized_constant_total_distance_m']:.3f} m",
                "Layer-specific improvement over best constant speed: "
                f"{comparison['layer_specific_improvement_m']:.3f} m "
                f"({comparison['layer_specific_improvement_percent']:.6f}%)",
                "",
                "The supplied layers.csv is marked illustrative; replace or justify its layer values before IA use.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return comparison


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--layers", type=Path, default=DEFAULT_LAYERS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    comparison = run(args.model, args.layers, args.output_dir)
    print(f"Layer-specific distance: {comparison['layer_specific_total_distance_m']:.3f} m")
    print(
        "Best constant-speed distance: "
        f"{comparison['optimized_constant_total_distance_m']:.3f} m at "
        f"{comparison['optimized_constant_airspeed_mps']:.6f} m/s"
    )
    print(
        "Improvement: "
        f"{comparison['layer_specific_improvement_m']:.3f} m "
        f"({comparison['layer_specific_improvement_percent']:.6f}%)"
    )
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
