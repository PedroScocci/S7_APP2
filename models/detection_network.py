import torch.nn as nn
import numpy as np
import torchvision

class AlexNetDectection(nn.Module):
    def __init__(self):
        super(AlexNetDectection, self).__init__()
        self.extraction = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
        )
        self.flatten = nn.Flatten(1,-1)
        self.localisation = nn.Sequential(
            nn.Linear(1936, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 7),
            nn.ReLU(inplace=True),
            nn.Linear(7, 4),
            nn.Sigmoid()
        )

    def create_model(self):
        model = nn.Sequential(self.extraction, self.flatten, self.localisation)
        return model

    def create_criterion(self, output, target):
        criterionBCE = nn.BCELoss()
        criterionMSE = nn.MSELoss()

        xyLoss = criterionBCE(output, target)
        whLoss = criterionMSE(output, target)
        criterion = 1*xyLoss + 4*whLoss + 0.5
        return 0