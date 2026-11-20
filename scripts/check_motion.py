"""Check a retargeted G1 motion against UniLab's robot model before training on it.

Reads a CSV in the Unitree convention (root x y z, root quaternion x y z w, then
the 29 joint angles), replays it on UniLab's G1 model, and reports how far the feet
are from the floor, joint-limit violations and joint speeds. Optionally writes a
copy shifted vertically so the supporting foot rests on the floor, and a preview
video.

The vertical shift is one height for the whole motion by default. With
--floor smooth it follows the slow drift in height that motions recovered from
video have, so the feet neither float nor sink in parts of the clip. Use that
only for motions without jumps. With --floor none the heights stay as they
are, which suits a motion recorded from the simulator (measure_tracking.py
--robot_csv writes one, at 50 rows per second).

Run from the UniLab folder so its environment is used:

    uv run --no-sync python /path/to/phone-video-to-g1/scripts/check_motion.py \
        --csv /path/to/my_motion_g1.csv \
        --out_csv /path/to/my_motion_g1_floor.csv \
        --preview /path/to/my_motion_preview.mp4
"""

import argparse
import pathlib
import re

import mujoco
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH

SCENE = ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"
FOOT_GEOM = re.compile(r"^(left|right)_foot\d+_collision$")


def load_csv(path: str) -> np.ndarray:
    """Return qpos per frame in MuJoCo layout: root xyz, root quaternion wxyz, joints."""
    rows = np.atleast_2d(np.loadtxt(path, delimiter=",", dtype=np.float64))
    if rows.shape[1] != 36:
        raise SystemExit(f"expected 36 columns (3 + 4 + 29), found {rows.shape[1]}")
    quat = rows[:, [6, 3, 4, 5]]
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    return np.concatenate([rows[:, :3], quat, rows[:, 7:]], axis=1)


def lowest_point(model: mujoco.MjModel, data: mujoco.MjData, geom_ids: list[int]) -> float:
    """Height of the lowest point of the given capsule geoms."""
    lowest = np.inf
    for g in geom_ids:
        radius, half_length = model.geom_size[g, 0], model.geom_size[g, 1]
        axis_z = data.geom_xmat[g].reshape(3, 3)[2, 2]
        lowest = min(lowest, data.geom_xpos[g, 2] - abs(axis_z) * half_length - radius)
    return float(lowest)


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Low-pass filter without time shift; the ends are padded with the edge values."""
    radius = int(np.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(values, radius, mode="edge"), kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", required=True, help="retargeted motion CSV")
    parser.add_argument("--fps", type=float, default=30.0, help="frame rate of the CSV")
    parser.add_argument(
        "--floor",
        choices=["constant", "smooth", "none"],
        default="constant",
        help="constant: one height shift for the whole motion; smooth: also remove slow drift; "
        "none: leave the heights as they are",
    )
    parser.add_argument(
        "--floor_window", type=float, default=0.5, help="smoothing time in seconds for --floor smooth"
    )
    parser.add_argument("--out_csv", help="write a floor-corrected copy here")
    parser.add_argument("--preview", help="write a preview video here (.mp4)")
    parser.add_argument("--reference_bodies", help="NPZ from retarget_to_g1.py --out_bodies")
    args = parser.parse_args()

    qpos = load_csv(args.csv)
    frames = qpos.shape[0]
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    if model.nq != 36:
        raise SystemExit(f"UniLab's G1 model has {model.nq} position values, expected 36")

    foot_geoms = {"left": [], "right": []}
    for g in range(model.ngeom):
        match = FOOT_GEOM.match(model.geom(g).name)
        if match and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            foot_geoms[match.group(1)].append(g)
    if not foot_geoms["left"] or not foot_geoms["right"]:
        raise SystemExit("could not find the foot collision capsules in UniLab's G1 model")

    hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    lower, upper = model.jnt_range[hinge, 0], model.jnt_range[hinge, 1]
    joint_names = [model.joint(j).name for j in hinge]

    clearance = np.zeros((frames, 2))
    body_pos = np.zeros((frames, model.nbody, 3))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        clearance[i] = [lowest_point(model, data, foot_geoms[s]) for s in ("left", "right")]
        body_pos[i] = data.xpos

    print(f"motion: {frames} frames at {args.fps:g} fps ({frames / args.fps:.1f} s)")
    travel = qpos[:, :2].max(axis=0) - qpos[:, :2].min(axis=0)
    print(f"root travels {travel[0]:.2f} m in x and {travel[1]:.2f} m in y")

    # At least one foot supports the robot in a motion without jumps, so the lower
    # foot's height over time estimates where the motion thinks the floor is.
    support = clearance.min(axis=1)
    floor_offset = float(np.median(support))
    residual = support - floor_offset
    print(f"floor offset (median height of the lower foot): {floor_offset * 1000:+.1f} mm")
    print(
        "after correcting it, the lower foot ranges from "
        f"{residual.min() * 1000:+.1f} mm to {residual.max() * 1000:+.1f} mm "
        f"(5th to 95th percentile: {np.percentile(residual, 5) * 1000:+.1f} to "
        f"{np.percentile(residual, 95) * 1000:+.1f} mm)"
    )

    joints = qpos[:, 7:]
    excess = np.maximum(lower - joints, 0.0) + np.maximum(joints - upper, 0.0)
    worst = int(excess.max(axis=0).argmax())
    violating = int((excess.max(axis=1) > 1e-3).sum())
    print(
        f"joint limits: {violating} of {frames} frames exceed a limit by more than 0.001 rad; "
        f"worst is {joint_names[worst]} by {excess.max():.3f} rad"
    )
    if frames > 1:
        speed = np.abs(np.diff(joints, axis=0)) * args.fps
        fastest = int(speed.max(axis=0).argmax())
        root_speed = np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1) * args.fps
        print(f"fastest joint: {joint_names[fastest]} at {speed.max():.1f} rad/s")
        print(f"fastest root speed: {root_speed.max():.2f} m/s")

    if args.reference_bodies:
        ref = np.load(args.reference_bodies)
        ref_names = [str(n) for n in ref["names"]]
        diffs = {}
        for b in range(1, model.nbody):
            name = model.body(b).name
            if name in ref_names:
                delta = body_pos[:, b] - ref["positions"][:, ref_names.index(name)]
                diffs[name] = float(np.linalg.norm(delta, axis=1).max())
        worst_body = max(diffs, key=diffs.get)
        print(
            f"UniLab's G1 vs GMR's G1 over {len(diffs)} shared bodies: "
            f"largest position difference {diffs[worst_body] * 1000:.2f} mm ({worst_body})"
        )

    if args.floor == "smooth":
        # A motion without jumps always has a foot on the floor, so the smoothed height
        # of the lower foot is the drift in the recovered height.
        correction = gaussian_smooth(support, args.floor_window * args.fps)
        residual = support - correction
        print(
            f"smooth floor correction ({args.floor_window:g} s): it varies from "
            f"{correction.min() * 1000:+.1f} to {correction.max() * 1000:+.1f} mm; afterwards the "
            f"lower foot ranges from {residual.min() * 1000:+.1f} mm to {residual.max() * 1000:+.1f} mm "
            f"(5th to 95th percentile: {np.percentile(residual, 5) * 1000:+.1f} to "
            f"{np.percentile(residual, 95) * 1000:+.1f} mm)"
        )
    elif args.floor == "none":
        correction = np.zeros(frames)
        print(
            "heights left as they are: the lower foot ranges from "
            f"{support.min() * 1000:+.1f} mm to {support.max() * 1000:+.1f} mm"
        )
    else:
        correction = np.full(frames, floor_offset)

    corrected = qpos.copy()
    corrected[:, 2] -= correction

    if args.out_csv:
        rows = np.concatenate(
            [corrected[:, :3], corrected[:, [4, 5, 6, 3]], corrected[:, 7:]], axis=1
        )
        out_csv = pathlib.Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_csv, rows, delimiter=",", fmt="%.8f")
        print(f"wrote the floor-corrected motion to {out_csv}")

    if args.preview:
        import imageio

        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance, camera.azimuth, camera.elevation = 3.2, 140.0, -12.0
        preview = pathlib.Path(args.preview)
        preview.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(preview, fps=args.fps) as writer:
            for q in corrected:
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                camera.lookat[:] = [q[0], q[1], 0.8]
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
        renderer.close()
        print(f"wrote the preview video to {preview}")


if __name__ == "__main__":
    main()
