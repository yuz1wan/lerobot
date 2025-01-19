HF_USER=$(huggingface-cli whoami | head -n 1)
export LD_LIBRARY_PATH=/home/wangziyu/anaconda3/envs/lerobot/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=1 python lerobot/scripts/train.py \
dataset_repo_id=${HF_USER}/so100_pour_cup \
policy=act_so100_real \
env=so100_real \
hydra.run.dir=outputs/train/act_so100_button \
hydra.job.name=act_so100_button \
device=cuda \
wandb.enable=true
