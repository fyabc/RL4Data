@rem cifar10 uncorrupted delta acc
python train.py G.job_name=@cifar10-reinforce-lr-delta_acc-NonC1@ G.action=@reload@ P.baseline_accuracy_file=@~/baseline_valacc_cifar10.txt@ epoch_per_episode=62

@rem cifar10 corrupted delta acc
python train.py G.job_name=@cifar10-reinforce-lr-delta_acc-Flip1@ G.action=@reload@ data_dir=@~/cifar10_flip.pkl.gz@ P.baseline_accuracy_file=@~/baseline_valacc_cifar10_flip.txt@ epoch_per_episode=62
