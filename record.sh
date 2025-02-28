HF_USER=$(huggingface-cli whoami | head -n 1)

python lerobot/scripts/control_robot.py record \
    --robot-path lerobot/configs/robot/so100.yaml \
    --fps 30 \
    --repo-id ${HF_USER}/so100_fold_0227_2 \
    --tags so100 fold \
    --warmup-time-s 5 \
    --episode-time-s 40 \
    --reset-time-s 5 \
    --num-episodes 15 \
    --push-to-hub 1 \
    --single-task "Fold the blue cloth diagonally."  \

