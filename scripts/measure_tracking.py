"""Record a trained UniLab motion-tracking policy and measure how well it follows the motion.

Runs UniLab's own playback, the code behind `uv run eval ... --render-mode record`, with
every robot starting at the first frame of the clip and playing it through once. While it
runs, this script logs for every robot and step which frame of the clip the robot is on,
whether UniLab reset it, and how far its joints, torso and tracked bodies are from the
reference. The playback video is written to the run folder as `play_video.mp4`, replacing
an earlier one, so the numbers describe that video. Nothing else in the run folder changes.

UniLab resets a robot the moment its torso, a foot or a hand drifts too far from the
reference, and puts it back at the start of the clip. A failure therefore never ends with
a robot lying on the floor in the video. The resets are counted here.

Errors use the pairing UniLab's rewards use: the robot's state after a step is compared
with the reference frame the policy was shown before that step. Training scores the torso's
position and orientation and the positions, orientations and speeds of the tracked bodies.
It does not score joint angles (that reward has weight 0), so the joint error is reported
for information only.

Run from the UniLab folder so its environment is used. The motion file must be a full path
without spaces:

    uv run --no-sync python /path/to/phone-video-to-g1/scripts/measure_tracking.py \
        --load_run 2026-11-19_22-23-32_mujoco \
        --motion_file /full/path/to/my_motion_g1.npz \
        --out /path/to/tracking_measurement.npz

Add `--no_video --robots 256` for a larger sample without rendering, and
`--robot_csv trained.csv` to write one robot's motion as a table that `check_motion.py`
can replay (`--fps 50 --floor none`). `--report_only` prints the report of an existing
measurement file again.

`--sim motrix` plays the policy in MotrixSim, a different physics engine from the MuJoCo it
was trained in. Everything else stays the same, so the two reports can be compared. MotrixSim
is an optional part of UniLab: `uv sync --frozen --extra mujoco --extra unirl --extra motrix`
adds it to an environment that was set up with the first two extras.

`--push 1` pushes every robot during the motion with UniLab's standard push. Its SAC and
FlashSAC tracking task files train with that push, and the PPO task file does not. Every 1 to
3 seconds the robot's velocity changes at once: by up to 0.5 m/s forwards and sideways, up to
0.2 m/s up or down, and by up to 0.52 rad/s of tilt and 0.78 rad/s of turn. Other numbers
scale all of these. The report then also counts the pushes and how many were followed by a
reset.
"""

import argparse
import pathlib
import runpy
import sys

import numpy as np

COMMAND_NAME = "motion"
CONTROL_RATE = 50.0  # control steps per second in UniLab's G1 motion tracking task
# For each simulator: the unisim module that holds its recording loop, and the loop's name.
SIMULATORS = {
    "mujoco": ("unisim.backend.mujoco.backend", "run_mujoco_playback"),
    "motrix": ("unisim.backend.motrix.backend", "run_motrix_playback"),
}
# UniLab's standard push for G1 motion tracking, the one its SAC and FlashSAC task files train
# with: a sudden change of the root's velocity, drawn for each axis between minus and plus
# these values (m/s for x, y, z in the world frame; rad/s for roll, pitch, yaw).
STANDARD_PUSH = {"x": 0.5, "y": 0.5, "z": 0.2, "roll": 0.52, "pitch": 0.52, "yaw": 0.78}
PUSH_WINDOW = 2.0  # a reset this many seconds or less after a push is counted to that push


def push_override(scale: float, every: tuple[float, float]) -> str:
    """The UniLab setting that adds velocity pushes of the given strength to the playback.

    Every robot keeps its own timer, so the pushes land at different moments of the clip.
    """
    ranges = ",".join(
        f"{axis}:[{-scale * limit:g},{scale * limit:g}]" for axis, limit in STANDARD_PUSH.items()
    )
    return (
        "+env.events.push_robot={func:unilab.envs.mdp.push_by_setting_velocity,mode:interval,"
        f"interval_range_s:[{every[0]:g},{every[1]:g}],is_global_time:false,"
        f"params:{{velocity_range:{{{ranges}}}}}}}"
    )


class Recorder:
    """Collects the robots' and the reference's state after the reset and after every step."""

    def __init__(self, save_robots: int) -> None:
        self.save_robots = save_robots
        self.rows: list[dict[str, np.ndarray]] = []
        self.pushes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.previous: dict | None = None
        self.meta: dict = {}
        self.max_relative_mismatch = 0.0

    def log_push(self, linear, angular, env_ids) -> None:
        """Remember a velocity push: which robots, how strong, and after which logged state.

        UniLab applies a push at the end of a step, so it changes the motion from the state
        that is logged next onwards.
        """
        import torch

        def rows(delta, count: int) -> np.ndarray:
            if delta is None:
                return np.zeros((count, 3), dtype=np.float32)
            return torch.as_tensor(delta).detach().cpu().numpy().reshape(count, 3).astype(np.float32)

        given = linear if linear is not None else angular
        count = int(torch.as_tensor(given).shape[0])
        if env_ids is None or isinstance(env_ids, slice):
            robots = np.arange(count)[env_ids if isinstance(env_ids, slice) else slice(None)]
        else:
            robots = torch.as_tensor(env_ids).detach().cpu().numpy().reshape(-1)
        state = np.full(len(robots), len(self.rows), dtype=np.int64)
        self.pushes.append((state, robots.astype(np.int64), rows(linear, count), rows(angular, count)))

    def log(self, env) -> None:
        import torch

        from unilab.utils.math import (
            quat_apply,
            quat_conjugate,
            quat_mul,
            yaw_from_quat,
            yaw_to_quat,
        )

        env = getattr(env, "unwrapped", env)
        command = env.command_manager.get_term(COMMAND_NAME)
        termination = env.termination_manager
        anchor = command.anchor_body_idx
        body_names = list(command.cfg.body_names)
        origins = env.scene.env_origins(command.time_steps.device)

        def reference_around_robot(ref_pos, ref_quat, robot_pos, robot_quat):
            """Reference body positions moved to the robot's torso position and heading.

            Same calculation as UniLab's `_update_motion_relative_state_torch`: heights stay
            those of the reference.
            """
            local = ref_pos - origins[:, None, :]
            turn = quat_mul(robot_quat[:, anchor], quat_conjugate(ref_quat[:, anchor]))
            heading = yaw_to_quat(0.5 * yaw_from_quat(turn))
            moved = quat_apply(heading[:, None, :], local - local[:, anchor][:, None, :])
            moved = moved + robot_pos[:, anchor][:, None, :]
            moved[..., 2] = local[..., 2]
            return moved

        frame = command.time_steps.to(torch.int64).clone()
        robot_joint = command.robot_joint_pos.clone()
        robot_pos = command.robot_body_pos_w.clone()
        robot_quat = command.robot_body_quat_w.clone()
        reference = {
            "frame": frame,
            "joint": command.joint_pos.clone(),
            "pos": command.body_pos_w.clone(),
            "quat": command.body_quat_w.clone(),
        }
        robots = frame.shape[0]

        # Check this script's alignment against the one UniLab computed for the current frame.
        mine = reference_around_robot(reference["pos"], reference["quat"], robot_pos, robot_quat)
        mismatch = float((mine - command.body_pos_relative_w).abs().max())
        self.max_relative_mismatch = max(self.max_relative_mismatch, mismatch)

        nan = torch.full((robots,), float("nan"))
        row = {
            "frame": frame,
            "valid": torch.zeros(robots, dtype=torch.bool),
            "terminated": torch.zeros(robots, dtype=torch.bool),
            "time_out": torch.zeros(robots, dtype=torch.bool),
            "joint_mean": nan,
            "joint_max": nan,
            "torso_pos": nan,
            "torso_xy": nan,
            "torso_z": nan,
            "torso_rot": nan,
            "body_mean": nan,
            "body_max": nan,
            "joint_by_joint": torch.full((robot_joint.shape[1],), float("nan")),
            "body_by_body": torch.full((len(body_names),), float("nan")),
        }
        for name in termination.active_terms:
            row[f"term_{name}"] = torch.zeros(robots, dtype=torch.bool)

        if self.previous is None:
            pelvis = body_names.index("pelvis")
            self.meta = {
                "body_names": np.array(body_names),
                "joint_names": np.array(list(command.robot.joint_names)),
                "termination_names": np.array(list(termination.active_terms)),
                "step_dt": float(env.step_dt),
                "anchor_body": body_names[anchor],
                "pelvis_index": pelvis,
                "initial_joint_offset_max": float((robot_joint - reference["joint"]).abs().max()),
            }
        else:
            before = self.previous
            # A robot that was reset in this step shows its new state here, and at the end of
            # the clip the reference returns to its first frame while the robot stays where
            # it is. Only robots that moved on by one frame are compared.
            valid = frame == before["frame"] + 1
            joint_error = (robot_joint - before["joint"]).abs()
            torso_error = robot_pos[:, anchor] - before["pos"][:, anchor]
            turn = quat_mul(quat_conjugate(before["quat"][:, anchor]), robot_quat[:, anchor])
            angle = 2.0 * torch.atan2(turn[:, 1:].norm(dim=-1), turn[:, 0].abs().clamp(max=1.0))
            moved = reference_around_robot(before["pos"], before["quat"], robot_pos, robot_quat)
            body_error = (moved - robot_pos).norm(dim=-1)

            def masked(values):
                return torch.where(valid, values, torch.full_like(values, float("nan")))

            row.update(
                valid=valid,
                terminated=termination.terminated.clone(),
                time_out=termination.time_outs.clone(),
                joint_mean=masked(joint_error.mean(dim=-1)),
                joint_max=masked(joint_error.max(dim=-1).values),
                torso_pos=masked(torso_error.norm(dim=-1)),
                torso_xy=masked(torso_error[:, :2].norm(dim=-1)),
                torso_z=masked(torso_error[:, 2].abs()),
                torso_rot=masked(angle),
                body_mean=masked(body_error.mean(dim=-1)),
                body_max=masked(body_error.max(dim=-1).values),
            )
            if bool(valid.any()):
                row["joint_by_joint"] = joint_error[valid].mean(dim=0)
                row["body_by_body"] = body_error[valid].mean(dim=0)
            for name in termination.active_terms:
                row[f"term_{name}"] = termination.get_term(name).clone()

        keep = min(self.save_robots, robots)
        pelvis = self.meta["pelvis_index"]
        row["root_pos"] = (robot_pos[:keep, pelvis] - origins[:keep]).clone()
        row["root_quat"] = robot_quat[:keep, pelvis].clone()
        row["root_lin_vel"] = command.robot_body_lin_vel_w[:keep, pelvis].clone()
        row["joint_pos"] = robot_joint[:keep].clone()

        self.rows.append({key: value.detach().cpu().numpy() for key, value in row.items()})
        self.previous = reference

    def arrays(self) -> dict[str, np.ndarray]:
        stacked = {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
        stacked.update({key: np.asarray(value) for key, value in self.meta.items()})
        stacked["relative_mismatch_max"] = np.asarray(self.max_relative_mismatch)
        if self.pushes:
            names = ("push_state", "push_robot", "push_linear", "push_angular")
            for name, parts in zip(names, zip(*self.pushes)):
                stacked[name] = np.concatenate(parts)
        return stacked


def run_playback(args: argparse.Namespace, frames: int) -> dict[str, np.ndarray]:
    """Run UniLab's eval entry point with logging added, and return the measurement."""
    import importlib

    from unilab import cli
    from unilab.base.entity import Entity
    from unilab.visualization import interactive_playback

    backend_module_name, record_name = SIMULATORS[args.sim]
    backend_module = importlib.import_module(backend_module_name)

    steps = args.steps if args.steps is not None else frames - 1
    overrides = [
        "training.sim2sim_strict=false",
        f"training.play_steps={steps}",
        # Longer than the playback, so no robot is reset for running out of time.
        f"env.max_episode_seconds={int(steps / CONTROL_RATE) + 2}",
        "env.commands.motion.params.sampling_mode=mixed",
        "env.commands.motion.params.sampling_start_ratio=1.0",
        f"env.commands.motion.params.motion_file={args.motion_file}",
    ]
    if args.robots is not None:
        overrides.append(f"training.play_env_num={args.robots}")
    if args.push:
        overrides.append(push_override(args.push, tuple(args.push_every)))
    overrides.extend(args.overrides)
    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="g1_motion_tracking",
        sim=args.sim,
        overrides=overrides,
        load_run=args.load_run,
        render_mode="record",
    )
    script, script_args = command[1], command[2:]

    recorder = Recorder(args.save_robots)
    session_class = interactive_playback.RslRlPlaybackSession
    original_reset, original_step = session_class.reset, session_class.step_once
    original_create = interactive_playback.create_rsl_rl_playback_session
    original_record = getattr(backend_module, record_name)
    original_push = Entity.apply_root_velocity_delta_to_sim
    checkpoint: list[str] = []

    def apply_push(self, linear_delta, angular_delta, env_ids=None, **push_kwargs):
        recorder.log_push(linear_delta, angular_delta, env_ids)
        return original_push(self, linear_delta, angular_delta, env_ids, **push_kwargs)

    def reset(self):
        obs = original_reset(self)
        recorder.log(self.env)
        return obs

    def step_once(self):
        obs = original_step(self)
        recorder.log(self.env)
        return obs

    def create_session(*create_args, **create_kwargs):
        result = original_create(*create_args, **create_kwargs)
        checkpoint.append(str(result[2]))
        runner = result[0].runner
        if runner is not None:
            # eval writes policy.onnx and policy.pt again on every run. A measurement leaves
            # the files in the run folder alone, apart from the video.
            runner.export_policy_to_onnx = lambda **_unused: None
            runner.export_policy_to_jit = lambda **_unused: None
        return result

    def step_without_video(*, initialize, step, num_steps, **_unused):
        """The stepping part of UniLab's recording loop, with the rendering left out."""
        obs = initialize()
        for _ in range(num_steps):
            obs = step(obs)
        return None

    session_class.reset, session_class.step_once = reset, step_once
    interactive_playback.create_rsl_rl_playback_session = create_session
    Entity.apply_root_velocity_delta_to_sim = apply_push
    if args.no_video:
        setattr(backend_module, record_name, step_without_video)
    saved_argv = sys.argv
    sys.argv = [script, *script_args]
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.argv = saved_argv
        session_class.reset, session_class.step_once = original_reset, original_step
        interactive_playback.create_rsl_rl_playback_session = original_create
        Entity.apply_root_velocity_delta_to_sim = original_push
        setattr(backend_module, record_name, original_record)

    if len(recorder.rows) != steps + 1:
        raise SystemExit(f"expected {steps + 1} logged states, got {len(recorder.rows)}")
    data = recorder.arrays()
    checkpoint_path = pathlib.Path(checkpoint[0])
    data["run"] = np.asarray(checkpoint_path.parent.name)
    data["checkpoint"] = np.asarray(checkpoint_path.name)
    data["motion_file"] = np.asarray(pathlib.Path(args.motion_file).name)
    data["motion_frames"] = np.asarray(frames)
    data["video"] = np.asarray(not args.no_video)
    data["sim"] = np.asarray(args.sim)
    if args.push:
        data["push_scale"] = np.asarray(float(args.push))
        data["push_every"] = np.asarray(args.push_every, dtype=float)
    return data


def pushes_followed_by_reset(data: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    """For every push, whether its robot was reset within PUSH_WINDOW seconds after it.

    A reset counts to the latest push before it. Also returns the number of resets with no
    push in the window before them.
    """
    window = int(round(PUSH_WINDOW / float(data["step_dt"])))
    state, robot = data["push_state"], data["push_robot"]
    followed = np.zeros(len(state), dtype=bool)
    by_robot: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index in np.unique(robot):
        mine = np.flatnonzero(robot == index)
        order = mine[np.argsort(state[mine], kind="stable")]
        by_robot[int(index)] = (state[order], order)
    unexplained = 0
    for step, index in np.argwhere(data["terminated"]):
        states, order = by_robot.get(int(index), (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)))
        # A push logged with state s acts from state s on, so it can cause a reset at s + 1 or later.
        latest = np.searchsorted(states, step, side="left") - 1
        if latest >= 0 and step - states[latest] <= window:
            followed[order[latest]] = True
        else:
            unexplained += 1
    return followed, unexplained


def print_report(data: dict[str, np.ndarray]) -> None:
    frame = data["frame"]
    states, robots = frame.shape
    steps = states - 1
    dt = float(data["step_dt"])
    seconds = steps * dt
    names = [str(name) for name in data["termination_names"]]
    failure_names = [name for name in names if name != "time_out"]
    deg, cm = np.degrees, 100.0

    print()
    print(
        f"run {data['run']}, {data['checkpoint']}, motion {data['motion_file']}, "
        f"played in {data.get('sim', 'mujoco')}"
    )
    print(
        f"{robots} robots, {steps} steps ({seconds:.2f} s); the clip has "
        f"{int(data['motion_frames'])} frames ({(int(data['motion_frames']) - 1) * dt:.2f} s)"
    )
    print(
        "checks: joints start within "
        f"{float(data['initial_joint_offset_max']):.3f} rad of the reference (UniLab scatters them "
        "by up to 0.1), and this script's torso alignment differs from UniLab's by at most "
        f"{float(data['relative_mismatch_max']) * 1000:.3f} mm"
    )

    terminated = data["terminated"]
    jumped = np.zeros_like(terminated)
    jumped[1:] = frame[1:] != frame[:-1] + 1
    unexplained = jumped & ~terminated & ~data["time_out"]
    reset_robots = np.flatnonzero(terminated.any(axis=0))
    print(f"played the whole {seconds:.2f} s without a reset: {robots - len(reset_robots)} of {robots} robots")
    events = np.argwhere(terminated)
    if len(events) == 0:
        print("resets for leaving the reference: none")
    else:
        print(f"resets for leaving the reference: {len(events)} in {len(reset_robots)} robots")
        for step, robot in events[:40]:
            reasons = [name for name in failure_names if data[f"term_{name}"][step, robot]]
            at = int(frame[step - 1, robot]) * dt
            print(f"  robot {robot}: reset at {at:.2f} s of the clip ({', '.join(reasons) or 'unknown'})")
        if len(events) > 40:
            print(f"  ... and {len(events) - 40} more")
        times = frame[events[:, 0] - 1, events[:, 1]] * dt
        counts, edges = np.histogram(times, bins=np.arange(0, np.ceil(seconds) + 1))
        busy = ", ".join(f"{int(lo)}-{int(lo) + 1} s: {n}" for lo, n in zip(edges[:-1], counts) if n)
        print(f"  resets by second of the clip: {busy}")
    if data["time_out"].any():
        print(f"time-outs: {int(data['time_out'].sum())}")
    if unexplained.any():
        print(f"robots that reached the end of the clip and started it again: {int(unexplained.sum())}")
    if "push_state" in data:
        followed, without_push = pushes_followed_by_reset(data)
        horizontal = np.linalg.norm(data["push_linear"][:, :2], axis=1)
        print(
            f"pushes: {len(followed)} in all, {len(followed) / robots:.1f} per robot on average; "
            f"horizontal speed change mean {horizontal.mean():.2f} m/s, largest {horizontal.max():.2f} m/s"
        )
        print(
            f"pushes followed by a reset within {PUSH_WINDOW:g} s: {int(followed.sum())} of {len(followed)} "
            f"({100 * followed.mean():.1f}%); resets with no push in the {PUSH_WINDOW:g} s before: {without_push}"
        )

    valid = data["valid"]
    if not valid.any():
        print("no compared steps")
        return
    print(f"compared robot-steps: {int(valid.sum())} of {steps * robots}")

    def line(label: str, values: np.ndarray, scale: float, unit: str) -> None:
        picked = values[valid] * scale
        worst = np.unravel_index(np.nanargmax(np.where(valid, values, np.nan)), values.shape)
        at = int(frame[worst[0] - 1, worst[1]]) * dt
        print(
            f"{label}: mean {picked.mean():.1f} {unit}, 95% of steps below "
            f"{np.percentile(picked, 95):.1f} {unit}, largest {picked.max():.1f} {unit} "
            f"(robot {worst[1]} at {at:.2f} s)"
        )

    # The training score is built from the first three groups. Joint angles are not part of
    # it (their reward weight is 0 in UniLab's G1 task), so they are listed last.
    bodies = len(data["body_names"])
    line(f"body position error with the torso aligned, averaged over the {bodies} tracked bodies", data["body_mean"], cm, "cm")
    line("largest single body position error with the torso aligned", data["body_max"], cm, "cm")
    line("torso distance from the reference path", data["torso_pos"], cm, "cm")
    line("  of which horizontal", data["torso_xy"], cm, "cm")
    line("  of which height", data["torso_z"], cm, "cm")
    line("torso orientation error", deg(data["torso_rot"]), 1.0, "deg")
    joints = data["joint_by_joint"].shape[1]
    line(f"joint angle error, averaged over the {joints} joints", deg(data["joint_mean"]), 1.0, "deg")
    line("largest single joint error", deg(data["joint_max"]), 1.0, "deg")

    weights = valid.sum(axis=1).astype(float)
    used = weights > 0
    body_by_body = np.average(data["body_by_body"][used], axis=0, weights=weights[used]) * cm
    order = np.argsort(body_by_body)[::-1]
    print(
        "bodies furthest from the reference (mean, torso aligned): "
        + ", ".join(f"{data['body_names'][i]} {body_by_body[i]:.1f} cm" for i in order[:4])
    )
    joint_by_joint = deg(np.average(data["joint_by_joint"][used], axis=0, weights=weights[used]))
    joint_names = data.get("joint_names", np.array([f"joint {i}" for i in range(joints)]))
    order = np.argsort(joint_by_joint)[::-1]
    print(
        "joints furthest from the reference (mean): "
        + ", ".join(f"{joint_names[i]} {joint_by_joint[i]:.1f} deg" for i in order[:4])
    )

    print("by second of the clip: body error (cm), horizontal torso distance (cm), joint error (deg)")
    clip_time = np.where(valid, np.roll(frame, 1, axis=0) * dt, np.nan)
    for second in range(int(np.ceil(seconds))):
        inside = valid & (clip_time >= second) & (clip_time < second + 1)
        if not inside.any():
            continue
        print(
            f"  {second:2d}-{second + 1:<2d} s: {data['body_mean'][inside].mean() * cm:5.1f}  "
            f"{data['torso_xy'][inside].mean() * cm:5.1f}  {deg(data['joint_mean'][inside]).mean():5.1f}"
        )


def write_robot_csv(data: dict[str, np.ndarray], path: str, robot: int | None) -> None:
    """Write one robot's motion in the table layout check_motion.py reads, 50 rows per second."""
    saved = data["root_pos"].shape[1]
    if robot is None:
        clean = np.flatnonzero(~data["terminated"][:, :saved].any(axis=0))
        if len(clean) == 0:
            raise SystemExit("every saved robot was reset; give --robot_index to write one anyway")
        robot = int(clean[0])
    if robot >= saved:
        raise SystemExit(f"only the first {saved} robots were saved; raise --save_robots")
    quat = data["root_quat"][:, robot]
    rows = np.concatenate(
        [data["root_pos"][:, robot], quat[:, [1, 2, 3, 0]], data["joint_pos"][:, robot]], axis=1
    )
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, rows, delimiter=",", fmt="%.8f")
    print(f"wrote robot {robot}'s motion ({rows.shape[0]} rows at {1 / float(data['step_dt']):g} per second) to {out}")
    if "push_state" in data:
        mine = np.flatnonzero(data["push_robot"] == robot)
        mine = mine[np.argsort(data["push_state"][mine], kind="stable")]
        dt = float(data["step_dt"])
        sizes = np.linalg.norm(data["push_linear"][mine, :2], axis=1)
        listed = ", ".join(f"{data['push_state'][i] * dt:.2f} s ({size:.2f} m/s)" for i, size in zip(mine, sizes))
        print(f"robot {robot} was pushed {len(mine)} times, with these horizontal parts: {listed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="measurement file to write (.npz)")
    parser.add_argument("--report_only", action="store_true", help="print the report of --out and stop")
    parser.add_argument("--motion_file", help="motion file the policy was trained on, as a full path without spaces")
    parser.add_argument("--load_run", default="-1", help="run folder name, or -1 for the newest run")
    parser.add_argument(
        "--sim",
        choices=sorted(SIMULATORS),
        default="mujoco",
        help="simulator to play the policy in (default: mujoco, the one it was trained in)",
    )
    parser.add_argument(
        "--push",
        type=float,
        help="push the robots during the motion: UniLab's standard push times this number "
        "(1 = up to 0.5 m/s forwards and sideways)",
    )
    parser.add_argument(
        "--push_every",
        type=float,
        nargs=2,
        default=[1.0, 3.0],
        metavar=("MIN", "MAX"),
        help="seconds between two pushes on the same robot (default: 1 to 3)",
    )
    parser.add_argument("--robots", type=int, help="number of robots (UniLab's playback default is 16)")
    parser.add_argument("--steps", type=int, help="steps to play (default: the clip once)")
    parser.add_argument("--no_video", action="store_true", help="measure without rendering a video")
    parser.add_argument("--save_robots", type=int, default=4, help="robots whose full motion is saved")
    parser.add_argument("--robot_csv", help="also write one robot's motion as a CSV for check_motion.py")
    parser.add_argument("--robot_index", type=int, help="robot for --robot_csv (default: first one never reset)")
    parser.add_argument("overrides", nargs="*", help="extra UniLab settings, for example training.cam_distance=4")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    if args.report_only:
        data = dict(np.load(out))
    else:
        if not args.motion_file:
            parser.error("--motion_file is required")
        motion = np.load(args.motion_file)
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        if abs(fps - CONTROL_RATE) > 1e-6:
            raise SystemExit(f"the motion file is {fps:g} frames per second; UniLab steps at {CONTROL_RATE:g}")
        data = run_playback(args, int(motion["joint_pos"].shape[0]))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **data)
        print(f"wrote the measurement to {out}")
    print_report(data)
    if args.robot_csv:
        write_robot_csv(data, args.robot_csv, args.robot_index)


if __name__ == "__main__":
    main()
