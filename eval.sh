HF_USER=$(huggingface-cli whoami | head -n 1)

python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/so100.yaml \
  --fps 30 \
  --repo-id ${HF_USER}/eval_act_so100_pickplace_0227 \
  --tags so100 tutorial eval \
  --warmup-time-s 5 \
  --episode-time-s 60 \
  --reset-time-s 3 \
  --num-episodes 20 \
  -p outputs/train/so100_pickplace_0223/checkpoints/last/pretrained_model \
  --single-task "Press the button."  \
  --policy-overrides device=mps
  