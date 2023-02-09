import argparse
import logging
import os
import torch
import numpy as np
from pathlib import Path

import evaluate
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

import mlflow
from transformers.integrations import MLflowCallback
from aim.hugging_face import AimCallback

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s->%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


class Solver:
    def __init__(self, args) -> None:
        self.args = args
        self.args.output_dir = Path(self.args.output_dir)

        self.experiment_name = f"xnli_{self.args.technique}"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
        )

    def run(self):
        for model_name in self.args.model_names:
            logging.info(f"solving XNLI using {model_name}")
            # dataset

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            train_dataset = (
                load_dataset("xnli", "en").shuffle(seed=self.args.seed)["train"]
            ).select(range(self.args.num_train_batches))
            train_dataset_tokenized = train_dataset.map(
                self.tokenize_function, batched=True
            )

            eval_datasets = {}
            for eval_dataset_name in self.args.eval_datasets_names:
                eval_dataset = load_dataset("xnli", eval_dataset_name).shuffle(
                    seed=self.args.seed
                )["validation"]
                eval_datasets[eval_dataset_name] = eval_dataset.map(
                    self.tokenize_function, batched=True
                )

            # model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=train_dataset.features["label"].num_classes
            ).to(self.device)

            if self.args.technique == "feature-extraction":
                for name, param in model.named_parameters():
                    # choose whatever you like here
                    if not name.startswith("classifier"):
                        param.requires_grad = False

            # metric
            self.metric = evaluate.load("accuracy")

            # trackers
            mlflow.set_tracking_uri(self.args.output_dir.joinpath("mlruns"))
            os.environ["MLFLOW_EXPERIMENT_NAME"] = self.experiment_name

            aim_callback = AimCallback(
                repo=str(self.args.output_dir.joinpath("aim_callback")),
                experiment=self.experiment_name,
            )

            # trainer
            training_args = TrainingArguments(
                output_dir=self.args.output_dir.joinpath("checkpoints"),
                evaluation_strategy="steps",
                report_to="mlflow",
                num_train_epochs=self.args.num_train_epochs,
                eval_steps=self.args.eval_steps,
                per_device_train_batch_size=self.args.batch_size,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset_tokenized,
                eval_dataset=eval_datasets,
                compute_metrics=self.compute_metrics,
                callbacks=[MLflowCallback, aim_callback],
            )

            trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve cross-lingual zero-shot transfer on XNLI dataset"
    )
    parser.add_argument(
        "technique",
        type=str,
        choices=["fine-tune", "feature-extraction"],
        help="technique to use",
    )
    parser.add_argument(
        "--model-names", nargs="+", help="names of the pre-trained models", default=[]
    )
    parser.add_argument(
        "--eval-datasets-names", nargs="+", help="subsets/languages of xnli", default=[]
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs",
        help="path to the directory to store the tracker repos and logs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="batch size",
        default=8,
    )
    parser.add_argument(
        "--num-train-batches",
        type=int,
        help="number of batches to train on",
        default=15000,
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        help="number of epochs to train",
        default=3.0,
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        help="after this many steps run the validation/evalutation",
        default=500,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="seed",
    )

    args = parser.parse_args()
    xnli_solver = Solver(args)
    xnli_solver.run()
