# aimlflow Examples

This folder contains a pre-made example script to quickly get started with.

## Example

1. Hyperparameter tuning - This example uses `pytorch` and `Ray Tune` to tune 3 hyperparameters, with 4 different batch sizes. `MLflow` is used as the parameter tracking tool. In order to start exploring the runs on `Aim UI` we convert runs into `Aim` storage and watch for new `MLflow` logs.
