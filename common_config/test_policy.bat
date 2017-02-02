@rem mnist corrupted test (for force init training jobs as example):
python train.py G.job_name=@mnist-stochastic-lr-ForceFlip1@ data_dir=@~/mnist_corrupted.pkl.gz@ epoch_per_episode=165 P.policy_load_file=@~/mnist-reinforce-lr-speed-ForceFlip1.npz@
