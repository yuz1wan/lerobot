HF_USER=$(huggingface-cli whoami | head -n 1)

python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/so100.yaml \
  --fps 30 \
  --repo-id ${HF_USER}/eval_act_so100_pp_pink \
  --tags so100 tutorial eval \
  --warmup-time-s 5 \
  --episode-time-s 40 \
  --reset-time-s 10 \
  --num-episodes 10 \
  -p outputs/train/act_so100_pp_pink/checkpoints/last/pretrained_model \
  --single-task "Press the button."  \