import argparse
import logging
import time
from pathlib import Path
from typing import List

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    has_method,
    init_keyboard_listener,
    init_policy,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.MiniTeleop_utils import LeadArmReader
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import init_hydra_config, init_logging, log_say, none_or_int


@safe_disconnect
def record(
    robot: Robot,
    root: Path,
    repo_id: str,
    single_task: str,
    fps: int | None = None,
    warmup_time_s: int | float = 2,
    episode_time_s: int | float = 10,
    reset_time_s: int | float = 5,
    num_episodes: int = 50,
    video: bool = True,
    run_compute_stats: bool = True,
    push_to_hub: bool = True,
    tags: list[str] | None = None,
    num_image_writer_processes: int = 0,
    num_image_writer_threads_per_camera: int = 4,
    display_cameras: bool = True,
    play_sounds: bool = True,
    resume: bool = False,
    # TODO(rcadene, aliberts): remove local_files_only when refactor with dataset as argument
    local_files_only: bool = False,
) -> LeRobotDataset:
    # TODO(rcadene): Add option to record logs
    listener = None
    events = None
    policy = None
    device = None
    use_amp = None

    if single_task:
        task = single_task
    else:
        raise NotImplementedError(
            "Only single-task recording is supported for now")

    if resume:
        dataset = LeRobotDataset(
            repo_id,
            root=root,
            local_files_only=local_files_only,
        )
        dataset.start_image_writer(
            num_processes=num_image_writer_processes,
            num_threads=num_image_writer_threads_per_camera *
            len(robot.cameras),
        )
        sanity_check_dataset_robot_compatibility(dataset, robot, fps, video)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(repo_id, policy)
        dataset = LeRobotDataset.create(
            repo_id,
            fps,
            root=root,
            robot=robot,
            use_videos=video,
            image_writer_processes=num_image_writer_processes,
            image_writer_threads=num_image_writer_threads_per_camera *
            len(robot.cameras),
        )

    if not robot.is_connected:
        robot.connect()

    listener, events = init_keyboard_listener()

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
    enable_teleoperation = policy is None
    log_say("Warmup record", play_sounds)
    warmup_record(robot, events, enable_teleoperation,
                  warmup_time_s, display_cameras, fps)

    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    recorded_episodes = 0
    while True:
        if recorded_episodes >= num_episodes:
            break

        # TODO(aliberts): add task prompt for multitask here. Might need to temporarily disable event if
        # input() messes with them.
        # if multi_task:
        #     task = input("Enter your task description: ")
        # print(">>>>>> breakpoint 1 <<<<<<<<")
        # time.sleep(3)

        robot.set_robot_home_position()

        log_say(f"Recording episode {dataset.num_episodes}", play_sounds)
        record_episode(
            dataset=dataset,
            robot=robot,
            events=events,
            episode_time_s=episode_time_s,
            display_cameras=display_cameras,
            policy=policy,
            device=device,
            use_amp=use_amp,
            fps=fps,
        )
        # print(">>>>>> breakpoint 2 <<<<<<<<")
        # time.sleep(3)

        # Execute a few seconds without recording to give time to manually reset the environment
        # Current code logic doesn't allow to teleoperate during this time.
        # TODO(rcadene): add an option to enable teleoperation during reset
        # Skip reset for the last episode to be recorded
        if not events["stop_recording"] and (
            (dataset.num_episodes < num_episodes -
             1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", play_sounds)
            robot.set_robot_home_position()
            reset_environment(robot, events, reset_time_s)
            # robot.set_robot_home_position()

        # print(">>>>>> breakpoint 3 <<<<<<<<")
        # time.sleep(3)

        if events["rerecord_episode"]:
            log_say("Re-record episode", play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # print(">>>>>> breakpoint 4 <<<<<<<<")
        # time.sleep(3)

        dataset.save_episode(task)
        recorded_episodes += 1

        # print(">>>>>> breakpoint 5 <<<<<<<<")
        # time.sleep(3)

        if events["stop_recording"]:
            break

    log_say("Stop recording", play_sounds, blocking=True)
    stop_recording(robot, listener, display_cameras)

    if run_compute_stats:
        logging.info("Computing dataset statistics")

    dataset.consolidate(run_compute_stats)

    if push_to_hub:
        dataset.push_to_hub(tags=tags)

    log_say("Exiting", play_sounds)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record a teleoperation session.")
    parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    parser.add_argument(
        "--robot-overrides",
        type=str,
        default=None,
        help="Path to robot overrides json file used to instantiate the robot using `make_robot` factory function.",
    )
    parser.add_argument(
        "--single-task",
        type=str,
        default="Task description",
        help="A short but accurate description of the task performed during the recording.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory where the dataset will be stored (e.g. 'dataset/path').",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser.add_argument(
        "--local-files-only",
        type=int,
        default=0,
        help="Use local files only. By default, this script will try to fetch the dataset from the hub if it exists.",
    )
    parser.add_argument(
        "--warmup-time-s",
        type=int,
        default=10,
        help="Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.",
    )
    parser.add_argument(
        "--episode-time-s",
        type=int,
        default=60,
        help="Number of seconds for data recording for each episode.",
    )
    parser.add_argument(
        "--reset-time-s",
        type=int,
        default=60,
        help="Number of seconds for resetting the environment after each episode.",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=50, help="Number of episodes to record.")
    parser.add_argument(
        "--run-compute-stats",
        type=int,
        default=1,
        help="By default, run the computation of the data statistics at the end of data collection. Compute intensive and not required to just replay an episode.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=int,
        default=1,
        help="Upload dataset to Hugging Face hub.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Add tags to your dataset on the hub.",
    )
    parser.add_argument(
        "--num-image-writer-processes",
        type=int,
        default=0,
        help=(
            "Number of subprocesses handling the saving of frames as PNGs. Set to 0 to use threads only; "
            "set to ≥1 to use subprocesses, each using threads to write images. The best number of processes "
            "and threads depends on your system. We recommend 4 threads per camera with 0 processes. "
            "If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses."
        ),
    )
    parser.add_argument(
        "--num-image-writer-threads-per-camera",
        type=int,
        default=4,
        help=(
            "Number of threads writing the frames as png images on disk, per camera. "
            "Too many threads might cause unstable teleoperation fps due to main thread being blocked. "
            "Not enough threads might cause low camera fps."
        ),
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Resume recording on an existing dataset.",
    )
    parser.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        type=str,
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`."
        ),
    )
    parser.add_argument(
        "--policy-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )

    args = parser.parse_args()

    init_logging()

    control_mode = args.mode
    robot_path = args.robot_path
    robot_overrides = args.robot_overrides
    kwargs = vars(args)
    del kwargs["mode"]
    del kwargs["robot_path"]
    del kwargs["robot_overrides"]

    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)

    record(robot, **kwargs)
