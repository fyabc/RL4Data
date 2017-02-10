@rem mnist corrupted test (for force init training jobs as example):
python train.py G.job_name=@mnist-stochastic-lr-ForceFlip1@ data_dir=@~/mnist_corrupted.pkl.gz@ epoch_per_episode=165 P.policy_load_file=@~/mnist-reinforce-lr-speed-ForceFlip1.npz@

@rem mnist random drop
python train.py G.job_name=@mnist-random_drop-speed-NonC3@ epoch_per_episode=165 P.random_drop_number_file=@~/drop_num_speed_NonC3.txt@
