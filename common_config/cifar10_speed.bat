@rem Default values of some options:
@rem model_name=@resnet@
@rem python train.py G.dataset=@cifar10@ G.train_type=@reinforce@ G.action=@reload@ P.policy_load_file=@~/C_speed.npz@ P.policy_save_file=@~/C_speed.npz@ epoch_per_episode=60 G.logging_file=@~/log_C_speed.txt@

@rem cifar10 uncorrupted speed:
python train.py G.job_name=@cifar10-reinforce-lr-speed-NonC1@ G.action=@reload@ epoch_per_episode=62 P.speed_reward_config=[[0.80,0.166667],[0.84,0.333333],[0.865,0.5]]

@rem cifar10 corrupted speed:
python train.py G.job_name=@cifar10-reinforce-lr-speed-Flip1@ G.action=@reload@ epoch_per_episode=62 data_dir=@~/cifar10_flip.pkl.gz@ P.speed_reward_config=[[0.72,0.166667],[0.76,0.333333],[0.795,0.5]]