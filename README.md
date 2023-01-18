<div align="center">
  <h1>aimlflow</h1>
  <h3>Aim-powered supercharged UI for MLFlow logs</h3>
  Run beautiful UI on top of your MLflow logs and get powerful run comparison features.
</div>

<br/>

<div align="center">

  [![Platform Support](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-blue)]()
  [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/aim-mlflow)](https://pypi.org/project/aim-mlflow/)
  [![PyPI Package](https://img.shields.io/pypi/v/aim-mlflow?color=yellow)](https://pypi.org/project/aim-mlflow/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)
  
</div>

<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/13848158/212019426-c60f2037-0faa-44f2-9620-88ab82c19f0a.png" />
</div>

## About

aimlflow helps to explore various types of metadata tracked during the training with MLFLow, including:

- hyper-parameters
- metrics
- images
- audio
- text

More about Aim: https://github.com/aimhubio/aim

More about MLFLow: https://github.com/mlflow/mlflow

## Getting Started

Follow the steps below to set up aimlflow.

1. Install aimlflow on your training environment:

```
pip install aim-mlflow
```

2. Run live time convertor to sync MLFlow logs with Aim:

```
aimlflow sync --mlflow-tracking-uri={mlflow_uri} --aim-repo={aim_repo_path}
```

3. Run the Aim UI:

```
aim up --repo={aim_repo_path}
```

## Why use aimlflow?

1. Powerful pythonic search to select the runs you want to analyze.

![image](https://user-images.githubusercontent.com/13848158/212019287-8c7a538c-d544-4b48-8e2a-9d3d2f90adbf.png)

2. Group metrics by hyperparameters to analyze hyperparameters’ influence on run performance.

![image](https://user-images.githubusercontent.com/13848158/212019346-a94c9fde-b1d1-4bcc-94ec-475ba7cebe75.png)

3. Select multiple metrics and analyze them side by side.

![image](https://user-images.githubusercontent.com/13848158/212019426-c60f2037-0faa-44f2-9620-88ab82c19f0a.png)

4. Aggregate metrics by std.dev, std.err, conf.interval.

![image](https://user-images.githubusercontent.com/13848158/212019455-3b607737-598b-4406-ac50-9b4317d37d16.png)

5. Align x axis by any other metric.

![image](https://user-images.githubusercontent.com/13848158/212019482-2e329f3b-b3ec-425e-a34f-e6f4e8464901.png)
 
6. Scatter plots to learn correlations and trends.

![image](https://user-images.githubusercontent.com/13848158/212019507-ae26cfc1-4a45-4233-a7ea-c503ead3dfd6.png)
 
7. High dimensional data visualization via parallel coordinate plot.

![image](https://user-images.githubusercontent.com/13848158/212019543-a6f70fba-2418-429b-911a-14bc250db33d.png)
