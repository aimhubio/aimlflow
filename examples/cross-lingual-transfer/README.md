# Zero-Shot Cross-Lingual transfer on XNLI dataset


Here we provide a demonstration on how to tackle the Zero-Shot Cross-Lingual transfer task using HuggingFace Transformers and Datasets. Afterwards, we'll keep track of the metadata utilizing both Aim and MLflow.

> It is important to keep in mind that due to limited computational resources, we have chosen to use only 15,000 samples for the training dataset and will be training for a mere 3 epochs with a batch size of 8. Consequently, the obtained results may not be optimal. Nevertheless, the purpose of this example is not to achieve the best possible results, but rather to showcase the advantages of using both Aim and MLflow. One can simply remove the training dataset samples limitation and train for longer and with bigger batch size.

## Explanation of the code
Let’s decompose each piece of the `main.py` script. First we will discuss the CLI arguments.

The first argument is `technique` which simply will indicate whether we want to fine tune the entire network: `fine-tune`, or to train only the classification head: `feature-extraction`.

Next mandatory arguments are the `--model-names` and the `--eval-datasets` which are list of names of pre-trained models and evaluation datasets respectively. As we are using `XNLI` dataset the later argument needs to simply be the names of the subsets(languages) of the `XNLI`. The `--output-dir` is a string which indicates where to store all of the outputs, i.e. `checkpoints`, `aim` and `mlflow` repositories. By default is set to `logs`The rest of the arguments tweak the parameters that their names suggest.

After successfully obtaining the specified arguments its time to pass the arguments to the `Solver`. The later contains the constructor, two helper methods and the main method `run` which fires the training process.

The constructor takes the arguments parsed by the `argparse`, sets the experiment name and the available device, if a gpu is available the device will be `cuda` otherwise it will fall back on `cpu`.

The `compute_metrics` method simply gets the logits of the output layer, the gold labels and computes the desired metric, which in our case is `accuracy`. Meanwhile `tokenize_function` will simply tokenize a given batch of samples, by truncating and padding the produced sequence to the length of the longest sequence in the batch.

The `run` method starts off with a loop, iterating over the pre-trained models names. By each `model_name` it initializes the corresponding tokenizer from the pre-trained counterpart, and loads the training dataset. After which it simply maps each batch of the training dataset to the tokenizer function, which produces the tokenized version of our dataset.

Essentially the same thing is done for evaluation datasets, for which we define a dictionary to keep the subpart name to `Dataset`(`load_dataset` method produces an instance this type) pairs.

Utilizing the `AutoModelForSequenceClassification` we load each pre-trained model by the specified `model_name` and load it on the available device. After loading the model we check wether the desired technique of training is `feature-extraction`, if thats the case we unfreeze the classification layer only.

Next we load accuracy metric from `evaluate` library, which will be used in the `compute_metrics` method.

After this we setup the trackers. For both MLflow and Aim we simply set the tracking repository path by prepending our output directory path, and the experiment name. For Aim we initialize an `AimCallback` meanwhile MLflow uses environment variables. The repo of `AimCallback` is initialized under `aim_callback` directory.

And the last steps before starting the training is to initialize a `TrainingArguments` and a `Trainer` with specified arguments. After which we can simply run the training 🚀.


## Usage example
An example of using the CLI for running trainings using the `feature-extraction` technique:

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
