#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import time
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import json
import datasets
import numpy as np
import torch.nn as nn
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from attack_methods import Attack_PGD
import torch.optim as optim
from tqdm import tqdm

import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    ViTFeatureExtractor,
    ViTForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import DataLoader
from vit_get_patch import ViTForImageClassificationGetpatch
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import pudb

""" Fine-tuning a 🤗 Transformers model for image classification"""

logger = logging.getLogger(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.15.0.dev0")

#require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

task_config = json.load(open("/home/LAB/hemr/workspace/Vit_position_attack/config/task_config.json", "r", encoding="utf-8"))

def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="nateraw/image-folder", metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        data_files = dict()
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

@dataclass
class TestArguments:
    test_type: str = field(
        default="normal",
        metadata={"help": "normal random withoutPE"},
    )
    log_step: int = field(
        default=7, 
        metadata={"help":'log_step'},
    )
    attack_model_dir: str = field(
        default="../model/attack_cifar10_vit/",
        metadata={"help": 'attack_model_dir'},
    )

def collate_fn(examples):
    #pixel_values = torch.stack([torch.tensor(example["pixel_values"]) for example in examples])   #[batch_size, 3, 224, 224]
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])

    #print(pixel_values)
    return pixel_values, labels

config_pgd = {
    'train': False,
    'targeted': False,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 20,
    'step_size': 2.0 / 255 * 2,
    'random_start': True,
    'top_k' : 32,
    'patch_num': 14,
    'image_size': 224,
    'embedding_d': 768,
    'loss_func': torch.nn.CrossEntropyLoss(reduction='none')
}

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, TestArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, test_args= parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, test_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Initialize our dataset and prepare it for the 'image-classification' task.
    ds = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        data_files=data_args.data_files,
        cache_dir=model_args.cache_dir
    )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in ds.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = ds["train"].train_test_split(data_args.train_val_split)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    #print(ds)
    labels = ds["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    metric = datasets.load_metric("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = ViTForImageClassificationGetpatch.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Define torchvision transforms to be applied to each image.
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        #print(len(example_batch["img"]))
        example_batch["pixel_values"] = [_train_transforms(Image.fromarray(np.array(f).astype('uint8'), mode='RGB')) for f in example_batch["img"]]
        #example_batch["pixel_values"] = torch.Tensor(example_batch["img"])
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [_val_transforms(Image.fromarray(np.array(f).astype('uint8'), mode='RGB'))for f in example_batch["img"]]
        #example_batch["pixel_values"] = torch.Tensor(example_batch["img"])
        #print(example_batch)
        return example_batch
    # Write model card and (optionally) push to hub
    ds["validation"].set_transform(val_transforms)
    valloader = DataLoader(ds["validation"], batch_size=training_args.per_device_eval_batch_size, collate_fn=collate_fn)

    start_epoch = 0
    net = Attack_PGD(model, feature_extractor, config_pgd)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(),
                          lr=training_args.learning_rate,
                          momentum=0.9, #momentum
                          weight_decay=2e-4)#args.weight_decay)

    def test(epoch, net):
        #print("epoch: ",epoch)
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        iterator = tqdm(valloader, ncols=0, leave=False)
        #print(inputs, " \n",targets)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            start_time = time.time()
            optimizer.zero_grad()
            #targets = targets.to(device)  #[bacth_size]

            outputs, loss = net(inputs, targets, batch_idx=batch_idx)

            loss.backward()
            """
            for i in optimizer.param_groups[0]['params']:
                if i.requires_grad == True:
                    print(i.grad)
            """
            #pu.db
            optimizer.step()
            test_loss += loss.item()

            duration = time.time() - start_time

            _, predicted = outputs.max(1)
            batch_size = targets.size(0)
            total += batch_size
            correct_num = predicted.eq(targets).sum().item()
            correct += correct_num
            iterator.set_description(
                str(predicted.eq(targets).sum().item() / targets.size(0)))

            if batch_idx % test_args.log_step == 0:
                print(
                    "step %d, duration %.2f, test  acc %.2f, avg-acc %.2f, loss %.2f"
                    % (batch_idx, duration, 100. * correct_num / batch_size,
                    100. * correct / total, test_loss / total))
        if epoch >= 0:
            print('Saving latest @ epoch %s..' % (epoch))
            f_path = os.path.join(test_args.attack_model_dir, 'latest')
            state = {
                'net': net.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            if not os.path.isdir(test_args.attack_model_dir):
                os.mkdir(test_args.attack_model_dir)
            torch.save(state, f_path)
        acc = 100. * correct / total
        print('Val acc:', acc)
        return acc

    for epoch in range(start_epoch, int(training_args.num_train_epochs)):
        test(epoch, net)

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "image-classification",
        "dataset": data_args.dataset_name,
        "tags": ["image-classification"],
    }


if __name__ == "__main__":
    main()