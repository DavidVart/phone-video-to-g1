"""Summarize measurement files written by scripts/measure_tracking.py.

Each positional argument is LABEL=FILE[,FILE...]. Files under one label are pooled, so samples
of different sizes can be combined. For every label the summary gives the robots that played
the whole clip without a reset (with a 95% Wilson interval), when their first resets happened,
the mean tracking errors and, for runs with pushes, how many pushes were followed by a reset
within 2 seconds.

    python analysis/summarize_measurements.py --out results/summary.json \
        --phase "one leg=18.5-23.6" \
        --compare "first policy|push-trained policy" \
        "first policy=run_a.npz,run_b.npz" "push-trained policy=run_c.npz"

--phase NAME=START-END also reports pushes that landed in that stretch of the clip (seconds).
--compare A|B runs Fisher's exact test on robots that played the whole clip, and on robots
whose first reset came before and after the first 2.5 seconds.
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from measure_tracking import pushes_followed_by_reset  # noqa: E402

# UniLab starts every robot with a random push; failures from it show up in the first seconds.
EARLY_SECONDS = 2.5
ERRORS = {
    "body_position_error_cm": ("body_mean", 100.0),
    "torso_distance_cm": ("torso_pos", 100.0),
    "torso_horizontal_distance_cm": ("torso_xy", 100.0),
    "torso_height_error_cm": ("torso_z", 100.0),
    "torso_orientation_error_deg": ("torso_rot", 180.0 / math.pi),
    "joint_angle_error_deg": ("joint_mean", 180.0 / math.pi),
}


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for a proportion, in percent."""
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [round(100 * (centre - half), 2), round(100 * (centre + half), 2)]


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for the 2x2 table [[a, b], [c, d]]."""
    row1, row2, col1, total = a + b, c + d, a + c, a + b + c + d
    denominator = math.comb(total, col1)

    def probability(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / denominator

    observed = probability(a)
    low, high = max(0, col1 - row2), min(row1, col1)
    return sum(p for p in (probability(x) for x in range(low, high + 1)) if p <= observed * (1 + 1e-9))


def summarize(paths: list[str], phases: dict[str, tuple[float, float]]) -> dict:
    robots = clean = steps = 0
    first_resets: list[float] = []
    sums = dict.fromkeys(ERRORS, 0.0)
    pushes = followed = 0
    phase_counts = {name: [0, 0] for name in phases}
    sims, runs = set(), set()
    for path in paths:
        data = dict(np.load(path))
        frame, valid, terminated = data["frame"], data["valid"], data["terminated"]
        dt = float(data["step_dt"])
        sims.add(str(data.get("sim", "mujoco")))
        runs.add(str(data["run"]))
        robots += frame.shape[1]
        clean += int((~terminated.any(axis=0)).sum())
        first: dict[int, float] = {}
        for step, robot in np.argwhere(terminated):
            first.setdefault(int(robot), int(frame[step - 1, robot]) * dt)
        first_resets.extend(first.values())
        for name, (key, scale) in ERRORS.items():
            sums[name] += float(data[key][valid].astype(np.float64).sum()) * scale
        steps += int(valid.sum())
        if "push_state" in data:
            hit, _ = pushes_followed_by_reset(data)
            pushes += len(hit)
            followed += int(hit.sum())
            state = np.minimum(data["push_state"], frame.shape[0] - 1)
            clip_time = frame[state, data["push_robot"]] * dt
            for name, (start, end) in phases.items():
                inside = (clip_time >= start) & (clip_time < end)
                phase_counts[name][0] += int(inside.sum())
                phase_counts[name][1] += int(hit[inside].sum())
    first_resets_array = np.array(first_resets)
    summary = {
        "files": [f"{pathlib.Path(p).parent.name}/{pathlib.Path(p).name}" for p in paths],
        "runs": sorted(runs),
        "simulators": sorted(sims),
        "robots": robots,
        "whole_clip_without_reset": clean,
        "whole_clip_without_reset_percent": round(100 * clean / robots, 2),
        "whole_clip_without_reset_wilson95_percent": wilson(clean, robots),
        f"robots_first_reset_before_{EARLY_SECONDS:g}s": int((first_resets_array < EARLY_SECONDS).sum()),
        f"robots_first_reset_from_{EARLY_SECONDS:g}s": int((first_resets_array >= EARLY_SECONDS).sum()),
        "compared_robot_steps": steps,
    }
    summary.update({f"mean_{name}": round(total / steps, 3) for name, total in sums.items()})
    if pushes:
        summary["pushes"] = {
            "count": pushes,
            "per_robot": round(pushes / robots, 2),
            "followed_by_reset_within_2s": followed,
            "followed_by_reset_percent": round(100 * followed / pushes, 2),
        }
        for name, (count, hits) in phase_counts.items():
            summary["pushes"][f"during {name}"] = {
                "count": count,
                "followed_by_reset": hits,
                "followed_by_reset_percent": round(100 * hits / count, 2) if count else None,
            }
    return summary


def compare(a: dict, b: dict) -> dict:
    early, late = f"robots_first_reset_before_{EARLY_SECONDS:g}s", f"robots_first_reset_from_{EARLY_SECONDS:g}s"

    def test(key_a: int, key_b: int) -> float:
        return float(f"{fisher_two_sided(key_a, a['robots'] - key_a, key_b, b['robots'] - key_b):.3g}")

    return {
        "fisher_p_whole_clip_without_reset": test(a["whole_clip_without_reset"], b["whole_clip_without_reset"]),
        f"fisher_p_{early}": test(a[early], b[early]),
        f"fisher_p_{late}": test(a[late], b[late]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("groups", nargs="+", help="LABEL=FILE[,FILE...]")
    parser.add_argument("--out", help="JSON file to write")
    parser.add_argument("--phase", action="append", default=[], help="NAME=START-END in seconds of the clip")
    parser.add_argument("--compare", action="append", default=[], help="LABEL_A|LABEL_B")
    args = parser.parse_args()

    phases = {}
    for item in args.phase:
        name, span = item.split("=", 1)
        start, end = (float(x) for x in span.split("-"))
        phases[name] = (start, end)
    groups = {}
    for item in args.groups:
        label, files = item.split("=", 1)
        groups[label] = summarize(files.split(","), phases)
    comparisons = {}
    for item in args.compare:
        label_a, label_b = item.split("|", 1)
        comparisons[f"{label_a} vs {label_b}"] = compare(groups[label_a], groups[label_b])

    for label, s in groups.items():
        line = (
            f"{label}: {s['whole_clip_without_reset']} of {s['robots']} robots played the whole clip "
            f"({s['whole_clip_without_reset_percent']}%); body {s['mean_body_position_error_cm']} cm, "
            f"torso {s['mean_torso_distance_cm']} cm, orientation {s['mean_torso_orientation_error_deg']} deg"
        )
        if "pushes" in s:
            line += f"; {s['pushes']['followed_by_reset_percent']}% of {s['pushes']['count']} pushes followed by a reset"
        print(line)
    for name, result in comparisons.items():
        print(f"{name}: {result}")
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"groups": groups, "comparisons": comparisons}, open(out, "w"), indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
