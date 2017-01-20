@rem Common used options:
@rem data_dir=@./data/cifar10/cifar10.pkl.gz@
@rem one_file=True (if False, read the data directory 'cifar-10-batches-py'
python train.py G.dataset=@cifar10@ epoch_per_episode=60 G.logging_file=@./log/cifar10/log_C_raw.txt@
