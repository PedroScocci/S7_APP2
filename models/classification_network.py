import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

class AlexNetClassification(nn.Module):
    def __init__(self, num_class=3):
        super(AlexNetClassification, self).__init__()
        self.extraction = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
        )

        self.classification = nn.Sequential(
            nn.Linear(10 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 3),
            nn.Sigmoid()
        )


    def create_model(self):
        model = torchvision.models.AlexNet()

        model = nn.Sequential(self.extraction, self.classification)

        return model

    def create_criterion(self):
        criterion = nn.BCELoss()
        return criterion