## Notes in policy I/O

1. Policy model file format:

    Format: `{policy_name}.{iter_number}.npz`
    Example: `M_speed_MLP_flip.25.npz`

## To add more...

- [NOTE]: Remember to change the job name, or it will load the old model.

- Set L2 factor of policy network:
    add `P.l2_c=0.0` into parameters
    (see in *common_config/mnist_delta_acc.bat*)

- Set force init policy weights:
    add these into parameters:

    ```python
    P.W_init=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,0.,-2.,0.]
    P.b_init=2.0
    ```

    The `W[-4]` is margin weight, `W[-2]` is loss rank weight.
