from __future__ import division, print_function

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models


def set_model(model_name, model_features):
    num_class = len(model_features.classes)
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        num_rs = model.fc.in_features
        model.fc = nn.Linear(num_rs, num_class)
    elif model_name == "resnet18_drop":
        model = models.resnet18(pretrained=True)
        num_rs = model.fc.in_features
        model.fc = nn.Linear(num_rs, num_class)
        model.fc = nn.Sequential(nn.Dropout(0.5), model.fc)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
        num_rs = model.fc.in_features
        model.fc = nn.Linear(num_rs, num_class)
    elif model_name == "resnet101_drop":
        model = models.resnet101(pretrained=True)
        num_rs = model.fc.in_features
        model.fc = nn.Linear(num_rs, num_class)
        model.fc = nn.Sequential(nn.Dropout(0.5), model.fc)
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        num_rs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_rs, num_class)
    elif model_name == "squeezenet":
        model = models.squeezenet1_0(pretrained=True)
        num_rs = model.classifier[1].in_channels
        model.classifier[1] = nn.Conv2d(num_rs, num_class, kernel_size=1)
    elif model_name == "vgg16":
        model = models.vgg16_bn(pretrained=True)
        num_rs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_rs, num_class)
    elif model_name == "densenet":
        model = models.densenet161(pretrained=True)
        num_rs = model.classifier.in_features
        model.classifier = nn.Linear(num_rs, num_class)
    elif model_name == "googlenet":
        model = models.googlenet(pretrained=True)
        num_rs = model.fc.in_features
        model.fc = nn.Linear(num_rs, num_class)
    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x1_0(pretrained=True)
        num_rs = model.fc.in_features
        model.fc = nn.Linear(num_rs, num_class)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        num_rs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_rs, num_class)
    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_large(pretrained=True)
        num_rs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_rs, num_class)
    elif model_name == "resnext":
        model = models.resnext50_32x4d(pretrained=True)
        num_rs = model.fc.in_features
        model.fc = nn.Linear(num_rs, num_class)
    elif model_name == "mnasnet":
        model = models.mnasnet1_0(pretrained=True)
        num_rs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_rs, num_class)
    elif model_name == "efficientnetb0":
        model = models.efficientnet_b0(pretrained=True)
        num_rs = model.classifier[1].in_features
    elif model_name == "efficientnetb1":
        model = models.efficientnet_b1(pretrained=True)
        num_rs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_rs, num_class)
    elif model_name == "efficientnetb2":
        model = models.efficientnet_b2(pretrained=True)
        num_rs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_rs, num_class)
    elif model_name == "efficientnetb4":
        model = models.efficientnet_b4(pretrained=True)
        num_rs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_rs, num_class)
    elif model_name == "efficientnetb5":
        model = models.efficientnet_b5(pretrained=True)
        num_rs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_rs, num_class)
    elif model_name == "efficientnetb6":
        model = models.efficientnet_b6(pretrained=True)
        num_rs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_rs, num_class)
    elif model_name == "efficientnetb7":
        model = models.efficientnet_b7(pretrained=True)
        num_rs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_rs, num_class)
    elif model_name == "regnet":
        model = models.regnet_x_32gf(pretrained=True)
        num_rs = model.fc.in_features
        model.fc = nn.Linear(num_rs, num_class)
    else:
        ValueError(f"Invalid model name: {model_name}")
    return model


def set_optimizer(model, model_features):
    lr = model_features.lr
    optimizer_name = model_features.optimizer
    if optimizer_name == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr)
    elif optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr)
    elif optimizer_name == "SparseAdam":
        optimizer = optim.SparseAdam(model.parameters(), lr)
    elif optimizer_name == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr)
    elif optimizer_name == "ASGD":
        optimizer = optim.ASGD(model.parameters(), lr)
    elif optimizer_name == "LBFGS":
        optimizer = optim.LBFGS(model.parameters(), lr)
    elif optimizer_name == "NAdam":
        optimizer = optim.NAdam(model.parameters(), lr)
    elif optimizer_name == "RAdam":
        optimizer = optim.RAdam(model.parameters(), lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RAdam(model.parameters(), lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr)
    elif optimizer_name == "Rprop":
        optimizer = optim.Rprop(model.parameters(), lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
    else:
        ValueError(f"Invalid optimizer name: {optimizer_name}")
    return optimizer


def set_scheduler(optimizer, model_features):
    num_epochs = model_features.num_epochs
    scheduler_name = model_features.scheduler
    if scheduler_name == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif scheduler_name == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(num_epochs * 0.5), int(num_epochs * 0.75)],
            gamma=0.1,
        )
    elif scheduler_name == "ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.001)
    elif scheduler_name == "CyclicLR":
        scheduler = lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.001,
            max_lr=0.1,
            step_size_up=10,
            step_size_down=10,
            mode="triangular2",
        )
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=0.001
        )
    else:
        ValueError(f"Invalid scheduler name: {scheduler_name}")
    return scheduler
