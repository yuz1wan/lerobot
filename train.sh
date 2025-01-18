HF_USER=$(huggingface-cli whoami | head -n 1)

CUDA_VISIBLE_DEVICES=1 python lerobot/scripts/train.py \
	  dataset_repo_id=${HF_USER}/so100_pp_pink \
	    policy=act_so100_real \
	      env=so100_real \
	        hydra.run.dir=outputs/train/act_so100_pp_pink \
		  hydra.job.name=act_so100_pp_pink \
		    device=cuda \
		      wandb.enable=true
