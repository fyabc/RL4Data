@rem Default values of some options:
@rem model_name=@resnet@
python train.py G.dataset=@cifar10@ G.train_type=@reinforce@ G.action=@reload@ P.policy_load_file=@~/C_speed.npz@ P.policy_save_file=@~/C_speed.npz@ epoch_per_episode=60 G.logging_file=@~/log_C_speed.txt@
