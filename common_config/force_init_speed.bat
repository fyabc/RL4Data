@rem W_init:
@rem    W shape is (15,):
@rem    [
@rem    softmax output probability (10)             => 0.0
@rem    output probability of the label (1)         => 0.0
@rem    margin (1)                                  => X
@rem    average accuracy (1)                        => 0.0
@rem    loss rank (1)                               => Y
@rem    accepted data number (1)                    => 0.0
@rem    ]
@rem X & Y can be set in options
@rem
@rem b_init: 2.0

@rem c-mnist force init speed:
python train.py G.job_name=@mnist-reinforce-lr-speed-ForceFlip1@ G.action=@reload@ valid_freq=125 epoch_per_episode=165 P.W_init=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,100,0.,0.5,0.] P.b_init=2.0 G.temp_job=@log_data@ data_dir=@~/mnist_corrupted.pkl.gz@  P.speed_reward_config=[[0.89,0.166667],[0.92,0.333333],[0.94,0.5]]


@rem A result shortcut of c-mnist SPL baseline:
@rem
@rem Validate Point 181: Epoch 19 Iteration 48423 Batch 683 TotalBatch 22750
@rem Training Loss: [NotComputed]
@rem History Training Loss: 0.80238731404
@rem Validate Loss: 0.322286976367
@rem Validate accuracy: 0.910799995184
@rem Test Loss: [NotComputed]
@rem Test accuracy: [NotComputed]
@rem Number of accepted cases: 455000 of 968460 total
@rem [Log Data]
@rem Part  (total     2503): 0.129	0.128	0.104	0.102	0.082	0.072	0.082	0.085	0.098	0.117
@rem Whole (total   455013): 0.150	0.145	0.129	0.101	0.073	0.061	0.066	0.079	0.093	0.104

@rem [NOTE]: The ratio of first class in SPL reach the highest 0.155 at about 900000 total cases, then drop slowly.

@rem c-mnist SPL baseline:
python train.py G.job_name=@mnist-spl-ForceFlip1@ valid_freq=125 epoch_per_episode=165*2 G.temp_job=@log_data@ data_dir=@~/mnist_corrupted.pkl.gz@


@rem c-cifar10 force init speed:
python train.py G.job_name=@cifar10-reinforce-lr-speed-ForceFlip1@ G.action=@reload@ valid_freq=125