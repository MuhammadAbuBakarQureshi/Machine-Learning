
import os
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import models
from torchmetrics import Accuracy

from datetime import datetime

## Save the best model

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    model_name = model_name if (model_name.endswith(".pt") or model_name.endswith(".pth")) else model_name + ".pt"

    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)

## Get the training time 

def print_train_time(start, end, device):

    time = end - start

    day = 0
    hour = 0
    min = 0
    sec = time

    while sec >= 60:

        if hour >= 24:

            day += 1
            hour -= 24

        elif sec >= 3600:

            hour += 1
            sec -= 3600

        elif time >= 60:

            min += 1
            sec -= 60

    day_output = f"{day} day{'s' if day > 1 else ''} : "
    hour_output = f"{hour} hour{'s' if hour > 1 else ''} : "
    min_output = f"{min} min{'s' if min > 1 else ''} : "
    sec_output = f"{sec} sec{'s' if sec > 1 else ''}"

    time_output = f"{day_output if day != 0 else ''}{hour_output if hour != 0 else ''}{min_output if min != 0 else ''}{sec_output if sec != 0 else ''}"

    print(f"Total time taken on {device} is {time_output}")

## Create tensorboard writer

def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str=None):
    
    time_stamp = datetime.now().strftime('%d-%m-%Y')

    if extra:

        log_dir = os.path.join("runs", time_stamp, experiment_name, model_name, extra)
    else:

        log_dir = os.path.join("runs", time_stamp, experiment_name, model_name)

    writer = SummaryWriter(log_dir=log_dir)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")

    return writer

## Set seeds

def set_seeds(SEED):

    torch.manual_seed(SEED)

    torch.cuda.manual_seed(SEED)

## Create effentb0 model

def create_effnetb0():

    ## download model

    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    ## Freeze feature parameters

    for params in model.features.parameters():

        params.requires_grad = False

    ## Set seeds

    set_seeds(42)

    ## Alter classifier for your need

    model.classifier = nn.Sequential(

        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=3, bias=True)
    )
    
    ## Change model name

    model.name = 'effnetb0'
    print(f"[INFO] Created new {model.name} model")
    return model

## Create effnetb2  model

def create_effnetb2():

    ## download model

    weights = models.EfficientNet_B2_Weights.DEFAULT
    model = models.efficientnet_b2(weights=weights)

    ## Freeze feature parameters

    for params in model.features.parameters():

        params.requires_grad = False

    ## Set seeds

    set_seeds(42)

    ## Alter classifier for your need

    model.classifier = nn.Sequential(

        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=3, bias=True)
    )
    
    ## Change model name

    model.name = 'effnetb2'
    print(f"[INFO] Created new {model.name} model")
    return model

def accuracy_metrics(classes,
                     device: torch.device):

    train_accuracy_fn = Accuracy(task='multiclass', num_classes=len(classes)).to(device)
    test_accuracy_fn = Accuracy(task='multiclass', num_classes=len(classes)).to(device)

    return train_accuracy_fn, test_accuracy_fn