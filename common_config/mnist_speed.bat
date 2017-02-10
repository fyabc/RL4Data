@rem Default values of some options:
@rem [NOTE]: the action may be @reload@ or @overwrite@
@rem P.reward_checker=@speed@
@rem P.policy_model_type=@lr@

python train.py G.dataset=@mnist@ G.train_type=@reinforce@ G.action=@reload@ P.reward_checker=@speed@ P.policy_load_file=@~/M_speed.npz@ P.policy_save_file=@~/M_speed.npz@ valid_freq=250 epoch_per_episode=165 G.logging_file=@~/log_M_speed.txt@

@rem Equivalent call: mnist speed raw
@rem 'lr' & 'speed' can be omitted, will use the default value
python train.py G.job_name=@mnist-reinforce-lr-speed-NonC1@ G.action=@reload@ valid_freq=125 epoch_per_episode=165

@rem mnist speed flip
python train.py G.job_name=@mnist-reinforce-lr-speed-Flip1@ G.action=@reload@ data_dir=@~/mnist_corrupted.pkl.gz@ valid_freq=125 epoch_per_episode=165 P.speed_reward_config=[[0.89,0.166667],[0.92,0.333333],[0.94,0.5]]
