"""Usage: python3 visualize_floorplan.py --json my_optimizer_solutions.json """
#!/usr/bin/env python3

import argparse
import json
from json import JSONDecodeError
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


DEFAULT_JSON_NAME = "my_optimizer_solutions.json"
OUTPUT_DIR = Path("floorplan_solution")


def load_all_solutions(json_path: str):
    """Load all floorplan solutions from a solution JSON file.

    Supported formats:
      1. {"solutions": [{"test_id": ..., "positions": ...}, ...]}
      2. {"test_id": ..., "positions": [[x, y, w, h], ...]}
      3. {"positions": [[x, y, w, h], ...]}  # test_id defaults to 0
    """
    json_path = Path(json_path)

    try:
        with json_path.open("r") as f:
            data = json.load(f)
    except JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON file: {json_path}\n"
            f"JSON parser stopped at line {exc.lineno}, column {exc.colno}. "
            "Please make sure the solution JSON was completely written."
        ) from exc

    solutions = []

    if "solutions" in data:
        for idx, item in enumerate(data["solutions"]):
            if "test_id" not in item:
                raise ValueError(f"solutions[{idx}] is missing 'test_id'.")
            if "positions" not in item:
                raise ValueError(f"solutions[{idx}] is missing 'positions'.")

            solutions.append((int(item["test_id"]), item["positions"]))

    elif "positions" in data:
        # Single-testcase format. If no test_id is stored, use 0.
        solutions.append((int(data.get("test_id", 0)), data["positions"]))

    else:
        raise ValueError("JSON must contain either 'solutions' or 'positions'.")

    if not solutions:
        raise ValueError("No solutions found in the JSON file.")

    return solutions


def rects_overlap(a, b, eps=1e-6):
    """Return True if two rectangles have positive-area overlap."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
    overlap_y = min(ay + ah, by + bh) - max(ay, by)

    return overlap_x > eps and overlap_y > eps


def find_overlap_blocks(positions):
    """Return block indices involved in at least one overlap."""
    overlap_blocks = set()

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if rects_overlap(positions[i], positions[j]):
                overlap_blocks.add(i)
                overlap_blocks.add(j)

    return overlap_blocks


def visualize_floorplan(positions, output_path, title="Floorplan"):
    """Draw a floorplan and save it as a PNG image."""
    if not positions:
        raise ValueError("positions is empty.")

    overlap_blocks = find_overlap_blocks(positions)

    min_x = min(x for x, y, w, h in positions)
    min_y = min(y for x, y, w, h in positions)
    max_x = max(x + w for x, y, w, h in positions)
    max_y = max(y + h for x, y, w, h in positions)

    width = max_x - min_x
    height = max_y - min_y
    margin = max(width, height) * 0.05
    if margin == 0:
        margin = 1.0

    fig, ax = plt.subplots(figsize=(10, 10))

    for idx, (x, y, w, h) in enumerate(positions):
        is_overlap = idx in overlap_blocks

        facecolor = "tab:red" if is_overlap else f"C{idx % 10}"
        edgecolor = "red" if is_overlap else "black"
        linewidth = 2.0 if is_overlap else 1.0
        alpha = 0.45 if is_overlap else 0.35

        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=alpha,
        )

        ax.add_patch(rect)

        # Draw the block index at the rectangle center.
        ax.text(
            x + w / 2,
            y + h / 2,
            str(idx),
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    info = (
        f"blocks={len(positions)}, "
        f"bbox=({width:.2f} x {height:.2f}), "
        f"overlap_blocks={len(overlap_blocks)}"
    )
    ax.text(
        0.01,
        0.99,
        info,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {output_path}")


def resolve_json_path(json_arg):
    """Resolve the JSON path, allowing the default file to live beside this script."""
    json_path = Path(json_arg)
    if json_path.exists():
        return json_path

    script_dir_path = Path(__file__).resolve().parent / json_arg
    if script_dir_path.exists():
        return script_dir_path

    return json_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize every test case stored in a solution JSON file. "
            "Each testcase is saved as one PNG under floorplan_solution/."
        )
    )
    parser.add_argument(
        "--json",
        default=DEFAULT_JSON_NAME,
        help=f"Path to solution JSON file. Default: {DEFAULT_JSON_NAME}",
    )

    args = parser.parse_args()
    json_path = resolve_json_path(args.json)
    solutions = load_all_solutions(json_path)

    for test_id, positions in solutions:
        output_path = OUTPUT_DIR / f"floorplan_test_{test_id}.png"
        visualize_floorplan(
            positions=positions,
            output_path=output_path,
            title=f"Floorplan test_id={test_id}",
        )

    print(f"Generated {len(solutions)} image(s) in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
