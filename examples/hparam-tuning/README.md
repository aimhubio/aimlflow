# Hyperparameter Tuning using Ray Tune

Here we provide a demo for using the `MLflow` experiment tracker along with the `Ray Tune` tunning framework to tune hyperparameters of CIFAR-10 training.

We generated the logs using the following command

```
python tune.py
```

By default, the logged parameters and metrics are stored in the `mlruns` directory.

Initialize an empty `Aim` repository in your preferred location on the file system, by this command

```
aim init
```

To convert the runs into `Aim` format and watch for new logs simply use the following command

```
aimlflow convert --mlflow-tracking-uri={mlflow_uri} --aim-repo={aim_repo_path} --watch
```

After which we just need to run the `Aim UI`

```
aim up
```

New runs, parameters, and metrics will be automatically converted, stored, and presented to you in the UI.
