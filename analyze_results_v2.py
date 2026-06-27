"""Summarize evaluator JSON and break down soft violation sources.

Usage:
    python analyze_results.py my_optimizer_results.json

If the JSON already contains soft-breakdown fields, they are averaged directly.
Otherwise this script recomputes boundary, cluster, and MIB violations from the
saved positions and the local LiteTensorDataTest validation inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional


Position = tuple[float, float, float, float]


def _avg(rows: Iterable[dict], key: str) -> Optional[float]:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return mean(float(v) for v in values)


def _sum_truth(rows: Iterable[dict], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def _finite_positions(row: dict) -> Optional[list[Position]]:
    positions = row.get("positions")
    if not positions:
        return None
    out: list[Position] = []
    for rect in positions:
        if rect is None or len(rect) != 4:
            return None
        x, y, w, h = (float(v) for v in rect)
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            return None
        out.append((x, y, w, h))
    return out


def _candidate_data_roots(results_path: Path, explicit: Optional[Path]) -> list[Path]:
    roots: list[Path] = []
    if explicit is not None:
        roots.append(explicit)

    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    result_dir = results_path.resolve().parent
    for base in (result_dir, cwd, here):
        roots.extend([base, base.parent, base.parent.parent])

    # Useful for the original Windows workspace; harmless elsewhere.
    roots.append(Path(r"C:\Users\angel\FloorSet"))

    unique: list[Path] = []
    seen = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _find_validation_root(results_path: Path, explicit: Optional[Path]) -> Optional[Path]:
    for root in _candidate_data_roots(results_path, explicit):
        if (root / "LiteTensorDataTest").is_dir():
            return root
    return None


def _load_constraints(root: Path, test_id: int):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to recompute soft breakdown") from exc

    files: list[Path] = []
    for config_id in range(21, 121):
        config_dir = root / "LiteTensorDataTest" / f"config_{config_id}"
        for identifier in range(1, 11):
            path = config_dir / f"litedata_{identifier}.pth"
            if path.is_file():
                files.append(path)

    if test_id < 0 or test_id >= len(files):
        raise IndexError(
            f"test_id {test_id} is outside LiteTensorDataTest range 0..{len(files) - 1}"
        )

    data = torch.load(files[test_id], map_location="cpu")
    block_info = data[0][0]
    area_targets = block_info[:, 0]
    block_count = int((area_targets != -1).sum().item())
    return block_info[:block_count, 1:].detach().cpu()


def _interval_overlap(a0: float, a1: float, b0: float, b1: float, eps: float) -> float:
    return min(a1, b1) - max(a0, b0) - eps


def _rects_connected(a: Position, b: Position, eps: float = 1e-6) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    overlap_w = min(ax2, bx2) - max(ax, bx)
    overlap_h = min(ay2, by2) - max(ay, by)
    if overlap_w > eps and overlap_h > eps:
        return True

    vertical_touch = (
        abs(ax2 - bx) <= eps or abs(bx2 - ax) <= eps
    ) and _interval_overlap(ay, ay2, by, by2, eps) >= 0.0
    horizontal_touch = (
        abs(ay2 - by) <= eps or abs(by2 - ay) <= eps
    ) and _interval_overlap(ax, ax2, bx, bx2, eps) >= 0.0
    return vertical_touch or horizontal_touch


def _component_count(rects: list[Position]) -> int:
    if not rects:
        return 0
    parent = list(range(len(rects)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _rects_connected(rects[i], rects[j]):
                union(i, j)
    return len({find(i) for i in range(len(rects))})


def _recompute_soft_breakdown(row: dict, constraints) -> dict:
    positions = _finite_positions(row)
    if positions is None:
        return {}

    block_count = min(len(positions), int(constraints.shape[0]))
    positions = positions[:block_count]
    fixed = constraints[:block_count, 0]
    preplaced = constraints[:block_count, 1]
    mib = constraints[:block_count, 2]
    cluster = constraints[:block_count, 3]
    boundary = constraints[:block_count, 4]

    boundary_violations = 0
    cluster_violations = 0
    mib_violations = 0
    n_soft = int((boundary != 0).sum().item())

    max_mib = int(mib.max().item()) if mib.numel() else 0
    for group_id in range(1, max_mib + 1):
        members = [i for i in range(block_count) if int(mib[i].item()) == group_id]
        n_soft += max(0, len(members) - 1)
        shapes = {(round(positions[i][2], 4), round(positions[i][3], 4)) for i in members}
        mib_violations += max(0, len(shapes) - 1)

    max_cluster = int(cluster.max().item()) if cluster.numel() else 0
    for group_id in range(1, max_cluster + 1):
        members = [i for i in range(block_count) if int(cluster[i].item()) == group_id]
        n_soft += max(0, len(members) - 1)
        components = _component_count([positions[i] for i in members])
        cluster_violations += max(0, components - 1)

    if positions:
        x_min = min(x for x, y, w, h in positions)
        y_min = min(y for x, y, w, h in positions)
        x_max = max(x + w for x, y, w, h in positions)
        y_max = max(y + h for x, y, w, h in positions)
        eps = 1e-6
        for i, (x, y, w, h) in enumerate(positions):
            code = int(boundary[i].item())
            if code == 0:
                continue
            touches = {
                1: abs(x - x_min) < eps,
                2: abs(x + w - x_max) < eps,
                4: abs(y + h - y_max) < eps,
                8: abs(y - y_min) < eps,
            }
            if not all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
                boundary_violations += 1

    total = boundary_violations + cluster_violations + mib_violations
    denom = max(n_soft, 1)
    return {
        "boundary_violations": boundary_violations,
        "cluster_violations": cluster_violations,
        "grouping_violations": cluster_violations,
        "mib_violations": mib_violations,
        "total_soft_violations": total,
        "max_possible_soft_violations": n_soft,
        "boundary_violations_relative": boundary_violations / denom,
        "cluster_violations_relative": cluster_violations / denom,
        "mib_violations_relative": mib_violations / denom,
    }


def _augment_rows(rows: list[dict], results_path: Path, data_path: Optional[Path]) -> str:
    needed = [
        "boundary_violations",
        "cluster_violations",
        "mib_violations",
    ]
    if all(all(row.get(key) is not None for key in needed) for row in rows):
        return "json_fields"

    root = _find_validation_root(results_path, data_path)
    if root is None:
        return "unavailable_no_LiteTensorDataTest"

    for row in rows:
        if all(row.get(key) is not None for key in needed):
            continue
        try:
            constraints = _load_constraints(root, int(row["test_id"]))
            row.update(_recompute_soft_breakdown(row, constraints))
        except Exception as exc:
            row["soft_breakdown_error"] = str(exc)
    return f"recomputed_from_{root}"


def summarize(path: Path, data_path: Optional[Path] = None) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("test_results", []))
    source = _augment_rows(rows, path, data_path)

    return {
        "tests": len(rows),
        "feasible": _sum_truth(rows, "is_feasible"),
        "total_score": payload.get("total_score"),
        "avg_cost": _avg(rows, "cost"),
        "avg_hpwl_gap": _avg(rows, "hpwl_gap"),
        "avg_area_gap": _avg(rows, "area_gap"),
        "avg_soft_violation": _avg(rows, "violations_relative"),
        "avg_boundary_violations": _avg(rows, "boundary_violations"),
        "avg_cluster_violations": _avg(rows, "cluster_violations"),
        "avg_mib_violations": _avg(rows, "mib_violations"),
        "avg_total_soft_violations": _avg(rows, "total_soft_violations"),
        "avg_max_possible_soft_violations": _avg(rows, "max_possible_soft_violations"),
        "avg_boundary_violation_relative": _avg(rows, "boundary_violations_relative"),
        "avg_cluster_violation_relative": _avg(rows, "cluster_violations_relative"),
        "avg_mib_violation_relative": _avg(rows, "mib_violations_relative"),
        "capped_cases": sum(1 for row in rows if float(row.get("cost", 0.0)) >= 9.999999),
        "avg_runtime_seconds": _avg(rows, "runtime_seconds"),
        "soft_breakdown_source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="FloorSet root containing LiteTensorDataTest; auto-detected by default.",
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.results, args.data_path), indent=2))


if __name__ == "__main__":
    main()
