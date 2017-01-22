@rem The original valid_freq is 2500 (2500 * 20 = 50000 data, one valid point per epoch)
python train.py G.dataset=@mnist@ G.train_type=@raw@ valid_freq=250 epoch_per_episode=165 G.logging_file=@./log/mnist/log_M_raw.txt@

@rem mnist uncorrupted raw:
python train.py G.job_name=@mnist-raw-NonC1@ epoch_per_episode=400

@rem mnist corrupted raw:
python train.py G.job_name=@mnist-raw-Flip1@ data_dir=@~/mnist_corrupted.pkl.gz@ epoch_per_episode=400