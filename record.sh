HF_USER=$(huggingface-cli whoami | head -n 1)

python lerobot/scripts/control_robot.py record \
    --robot-path lerobot/configs/robot/so100.yaml \
    --fps 30 \
    --repo-id ${HF_USER}/so100_button \
    --tags so100 tutorial \
    --warmup-time-s 5 \
    --episode-time-s 40 \
    --reset-time-s 10 \
    --num-episodes 10 \
    --push-to-hub 1 \
    --single-task "Press the button."  \
