@rem Common used options:
@rem data_dir=@./data/cifar10/cifar10.pkl.gz@
@rem one_file=True (if False, it will read the data directory './data/cifar-10-batches-py')
python train.py G.dataset=@cifar10@ G.train_type=@raw@ epoch_per_episode=60 G.logging_file=@./log/cifar10/log_C_raw.txt@

@rem cifar10 uncorrupted raw:
python train.py G.job_name=@cifar10-raw-NonC1@ epoch_per_episode=82 valid_freq=390
