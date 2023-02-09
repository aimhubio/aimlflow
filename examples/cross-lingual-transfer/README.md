# Zero-Shot Cross-Lingual transfer using XNLI dataset


Here we provide a demonstration on how to tackle the Zero-Shot Cross-Lingual transfer task using HuggingFace Transformers and Datasets. Afterwards, we'll keep track of the metadata utilizing both Aim and MLflow.

An example of the CLI usage:

```
python main.py feature-extraction \
    --model-names bert-base-multilingual-cased bert-base-multilingual-uncased xlm-roberta-base distilbert-base-multilingual-cased \
    --eval-datasets-names en de fr es ar zh \
    --output-dir /PATH_TO/logs
```

In this particular command we use feature-extraction technique for 4 pre-trained models and validating on 6 languages of xnli dataset.

Initialize an empty `Aim` repository in your preferred location on the file system, by this command

```
aim init
```

To convert the runs into `Aim` format and watch for new logs simply use the following command

```
aimlflow sync --mlflow-tracking-uri={mlflow_uri} --aim-repo={aim_repo_path}
```

After which we just need to run the `Aim UI`

```
aim up
```

New runs, parameters, and metrics will be automatically converted, stored, and presented to you in the UI.
