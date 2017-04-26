# RL4Data

Using policy gradient to select training data of neural network.

## Dependency

[Theano](https://github.com/Theano/Theano)
[lasagne](https://github.com/Lasagne/Lasagne)

## Run

Command:

```bash

cd /path/to/project/root

python train.py [args...]

```

See [config.json](./config.json) to know how to set configurations.

1. Run baseline model (without policy)

`python train.py G.job_name=@cifar10-raw-Xxx@`

2. Run SPL algorithm

`python train.py G.job_name=@cifar10-spl-Xxx@`

2. Train policy with REINFORCE

`python train.py G.job_name=@cifar10-policy-lr-speed-Xxx@`

3. Train policy with Actor-Critic

`python train.py G.job_name=@cifar10-ac-lr-Xxx@`

4. Test policy

`python train.py G.job_name=@cifar10-stochastic-lr-speed-Xxx@ P.policy_load_file=@~/cifar10-policy-lr-speed-Xxx.4.npz@`

5. Random drop test

`python train.py G.job_name=@cifar10-random_drop-Xxx@ P.random_drop_number_file=@~/xxx.txt@`
