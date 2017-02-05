@rem mnist uncorrupted delta acc
python train.py G.job_name=@mnist-reinforce-lr-delta_acc-NonC1@ G.action=@reload@ P.baseline_accuracy_file=@~/baseline_valacc_mnist.txt@ valid_freq=125 epoch_per_episode=165

@rem mnist corrupted delta acc
python train.py G.job_name=@mnist-reinforce-lr-delta_acc-Flip1@ G.action=@reload@ P.baseline_accuracy_file=@~/baseline_valacc_mnist_flip.txt@ valid_freq=125 epoch_per_episode=165 data_dir=@~/mnist_corrupted.pkl.gz@

@rem mnist uncorrupted delta acc
python train.py G.job_name=@mnist-reinforce-lr-delta_acc-NonC1NoL2@ G.action=@reload@ P.baseline_accuracy_file=@~/baseline_valacc_mnist.txt@ valid_freq=125 epoch_per_episode=165 P.l2_c=0.0

@rem mnist corrupted delta acc with no L2 factor
python train.py G.job_name=@mnist-reinforce-lr-delta_acc-Flip1NoL2@ G.action=@reload@ P.baseline_accuracy_file=@~/baseline_valacc_mnist_flip.txt@ valid_freq=125 epoch_per_episode=165 data_dir=@~/mnist_corrupted.pkl.gz@ P.l2_c=0.0
