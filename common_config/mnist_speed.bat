@rem Default values of some options:
@rem [NOTE]: the action may be @reload@ or @overwrite@
@rem P.reward_checker=@speed@
@rem P.policy_model_name=@lr@

python train.py G.dataset=@mnist@ G.train_type=@reinforce@ G.action=@reload@ P.reward_checker=@speed@ P.policy_load_file=@~/M_speed.npz@ P.policy_save_file=@~/M_speed.npz@ valid_freq=250 epoch_per_episode=165 G.logging_file=@~/log_M_speed.txt@

@rem Equivalent call:
@rem 'lr' & 'speed' can be omitted, will use the default value
python train.py G.job_name=@mnist_reinforce_lr_speed_NonC1@ G.action=@reload@ valid_freq=250 epoch_per_episode=165
