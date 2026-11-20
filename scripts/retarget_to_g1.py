"""Retarget a human motion onto the Unitree G1 with GMR, without a viewer window.

Reads a GVHMR result file (hmr4d_results.pt) or, with --synthetic, GMR's built-in
marching test motion. Writes a headerless CSV in the Unitree convention that
UniLab's scripts/motion/csv_to_npz.py reads: root x y z, root quaternion x y z w,
then the 29 joint angles, at 30 rows per second.

GMR treats a GVHMR result as 30 frames per second, so the video given to GVHMR
must be 30 fps.

Run from the GMR folder so its environment is used:

    uv run --no-sync python /path/to/phone-video-to-g1/scripts/retarget_to_g1.py \
        --gvhmr_pred_file /path/to/hmr4d_results.pt \
        --out_csv /path/to/my_motion_g1.csv
"""

import argparse
import pathlib
import sys

import general_motion_retargeting
import mujoco
import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.params import ROBOT_XML_DICT

GMR_ROOT = pathlib.Path(general_motion_retargeting.__file__).resolve().parent.parent
BODY_MODELS = GMR_ROOT / "assets" / "body_models"
ROBOT = "unitree_g1"


def load_human_frames(args: argparse.Namespace) -> tuple[list[dict], float, float | None]:
    """Return per-frame human body targets, their frame rate, and the person's height if known."""
    if args.synthetic:
        sys.path.insert(0, str(GMR_ROOT / "tests"))
        from synthetic_motion import FPS, build_frames

        return [dict(frame) for frame in build_frames()], float(FPS), None

    from general_motion_retargeting.utils.smpl import (
        get_gvhmr_data_offline_fast,
        load_gvhmr_pred_file,
    )

    model_file = BODY_MODELS / "smplx" / "SMPLX_NEUTRAL.npz"
    if not model_file.exists():
        raise SystemExit(
            f"SMPL-X body model not found: {model_file}\n"
            "Download it from https://smpl-x.is.tue.mpg.de/ (free registration) and place "
            "SMPLX_NEUTRAL.npz in that folder."
        )
    smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(
        args.gvhmr_pred_file, BODY_MODELS
    )
    frames, fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
    return list(frames), float(fps), float(height)


def retarget(frames: list[dict], height: float | None, warmup: int) -> np.ndarray:
    """Return robot qpos per frame: root xyz, root quaternion wxyz, 29 joint angles."""
    retargeter = GeneralMotionRetargeting("smplx", ROBOT, actual_human_height=height, verbose=False)
    # Let the solver settle on the first pose, so frame 0 is as accurate as the rest.
    for _ in range(warmup):
        retargeter.retarget(dict(frames[0]))
    return np.stack([np.array(retargeter.retarget(dict(frame)), dtype=np.float64) for frame in frames])


def body_positions(qpos: np.ndarray) -> tuple[list[str], np.ndarray]:
    """World position of every named robot body per frame, from GMR's own G1 model."""
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[ROBOT]))
    data = mujoco.MjData(model)
    names = [model.body(b).name for b in range(1, model.nbody)]
    out = np.zeros((qpos.shape[0], len(names), 3))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        out[i] = data.xpos[1:]
    return names, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gvhmr_pred_file", help="hmr4d_results.pt written by GVHMR")
    source.add_argument("--synthetic", action="store_true", help="use GMR's built-in test motion")
    parser.add_argument("--out_csv", required=True, help="CSV to write")
    parser.add_argument("--out_bodies", help="optional NPZ of robot body positions, for cross-checks")
    parser.add_argument("--warmup", type=int, default=10, help="solver passes on the first frame")
    args = parser.parse_args()

    frames, fps, height = load_human_frames(args)
    print(f"human motion: {len(frames)} frames at {fps:g} fps ({len(frames) / fps:.1f} s)")
    if height is not None:
        print(f"estimated person height: {height:.2f} m")
    if abs(fps - 30.0) > 1e-6:
        raise SystemExit(f"expected 30 fps from GMR's loader, got {fps:g}")

    qpos = retarget(frames, height, args.warmup)
    if not np.isfinite(qpos).all():
        raise SystemExit("retargeting produced non-finite values")

    rows = np.concatenate([qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]], axis=1)
    out_csv = pathlib.Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, rows, delimiter=",", fmt="%.8f")
    print(f"wrote {rows.shape[0]} rows x {rows.shape[1]} columns to {out_csv}")
    print(f"root height range: {qpos[:, 2].min():.3f} to {qpos[:, 2].max():.3f} m")

    if args.out_bodies:
        names, positions = body_positions(qpos)
        np.savez(args.out_bodies, names=np.array(names), positions=positions, fps=fps)
        print(f"wrote body positions for {len(names)} bodies to {args.out_bodies}")


if __name__ == "__main__":
    main()
