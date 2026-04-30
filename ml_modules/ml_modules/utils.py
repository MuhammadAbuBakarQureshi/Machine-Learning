
import os
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import models
from torchmetrics import Accuracy
from torchinfo import summary

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
        nn.Linear(in_features=1280, out_features=6, bias=True)
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
        nn.Linear(in_features=1408, out_features=6, bias=True)
    )
    
    ## Change model name

    model.name = 'effnetb2'
    print(f"[INFO] Created new {model.name} model")
    return model

def create_resnet101(num_classes:int):

    resnet_101_weights = models.ResNet101_Weights.DEFAULT
    resnet_101_model = models.resnet101(weights=resnet_101_weights)

    ## Freeze pretrained layers

    for params in resnet_101_model.parameters():

        params.requires_grad = False

    ## Unfreeze output layer

    layers = ['layer4', 'avgpool', 'fc']

    for layer in layers:

        for params in getattr(resnet_101_model, layer).parameters():

            params.requires_grad = True

    ## Alter output layer

    resnet_101_model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    
    resnet_101_model.name = 'resnet101'
    print(f"[INFO] Created new {resnet_101_model.name} model")

    return resnet_101_model

# def create_vit_b_16(num_classes):

        
#     weights = models.ViT_B_16_Weights.DEFAULT
#     vit_b_16 = models.vit_b_16(weights=weights)

#     for params in vit_b_16.parameters():

#         params.requires_grad = False


#     vit_b_16.heads = nn.Sequential(
#                     nn.Linear(in_features=768,
#                             out_features=num_classes,
#                             bias=True)
#                 )
#     vit_b_16.name = "vit_b_16"
#     print(f"[INFO] Created new {vit_b_16.name} model")
#     return vit_b_16


def create_vit_b_16(num_classes: int, unfreeze_layers: int):

        
    weights = models.ViT_B_16_Weights.DEFAULT
    vit_b_16 = models.vit_b_16(weights=weights)

    for params in vit_b_16.parameters():

        params.requires_grad = False


    ## Unfreeze layers

    layers_to_unfreeze = [

        vit_b_16.encoder.ln,
        vit_b_16.encoder.layers.encoder_layer_11,
        vit_b_16.encoder.layers.encoder_layer_10,
        vit_b_16.encoder.layers.encoder_layer_9,
        vit_b_16.encoder.layers.encoder_layer_8,
        vit_b_16.encoder.layers.encoder_layer_7,
        vit_b_16.encoder.layers.encoder_layer_6,
        vit_b_16.encoder.layers.encoder_layer_5
    ]

    for i, layer in enumerate(layers_to_unfreeze):

        i += 1

        if unfreeze_layers >= i:

            for params in layer.parameters():

                params.requires_grad = True

        else:

            break


    vit_b_16.heads = nn.Sequential(
                    nn.Linear(in_features=768,
                            out_features=num_classes,
                            bias=True)
                )
    vit_b_16.name = "vit_b_16"
    print(f"[INFO] Created new {vit_b_16.name} model")
    return vit_b_16

def create_vit_b_32(num_classes: int, unfreeze_layers: int):

        
    weights = models.ViT_B_32_Weights.DEFAULT
    vit_b_32 = models.vit_b_32(weights=weights)

    for params in vit_b_32.parameters():

        params.requires_grad = False

    ## Unfreeze layers

    layers_to_unfreeze = [

        vit_b_32.encoder.ln,
        vit_b_32.encoder.layers.encoder_layer_11,
        vit_b_32.encoder.layers.encoder_layer_10,
        vit_b_32.encoder.layers.encoder_layer_9,
        vit_b_32.encoder.layers.encoder_layer_8,
        vit_b_32.encoder.layers.encoder_layer_7,
        vit_b_32.encoder.layers.encoder_layer_6,
        vit_b_32.encoder.layers.encoder_layer_5
    ]

    for i, layer in enumerate(layers_to_unfreeze):

        i += 1

        if unfreeze_layers >= i:

            for params in layer.parameters():

                params.requires_grad = True

        else:

            break


    vit_b_32.heads = nn.Sequential(
                    nn.Dropout(p=0.3),
                    nn.Linear(in_features=768,
                            out_features=num_classes,
                            bias=True)
                )
    vit_b_32.name = "vit_b_32"
    print(f"[INFO] Created new {vit_b_32.name} model")
    return vit_b_32

def create_vit_l_16(num_classes):

        
    weights = models.ViT_L_16_Weights.DEFAULT
    vit_l_16 = models.vit_l_16(weights=weights)

    for params in vit_l_16.parameters():

        params.requires_grad = False

    vit_l_16.heads = nn.Sequential(
                    nn.Linear(in_features=1024,
                            out_features=num_classes,
                            bias=True)
                )
    vit_l_16.name = "vit_l_16"
    print(f"[INFO] Created new {vit_l_16.name} model")
    return vit_l_16


def accuracy_metrics(classes,
                     device: torch.device):

    train_accuracy_fn = Accuracy(task='multiclass', num_classes=len(classes)).to(device)
    test_accuracy_fn = Accuracy(task='multiclass', num_classes=len(classes)).to(device)

    return train_accuracy_fn, test_accuracy_fn

def model_summary(model,
                  input_size=[32, 3, 244, 244]):

    print(summary(model=model,
        input_size=input_size,
        col_names=['num_params', 'input_size', 'output_size', 'trainable'], 
        col_width=20,
        row_settings=['var_names']))