import torch
import torch.nn as nn
import numpy as np
import torchvision

class AlexNetDectection(nn.Module):
    def __init__(self):
        super(AlexNetDectection, self).__init__()
        self.hidden = 16
        self.extraction = nn.Sequential(
            nn.Conv2d(1, 4*self.hidden, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(4*self.hidden),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(4*self.hidden, 6*self.hidden, kernel_size=5, padding=1, stride=2),
            nn.BatchNorm2d(6 * self.hidden),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(6*self.hidden, 6*self.hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(6 * self.hidden),
            nn.ReLU(),
            nn.Conv2d(6*self.hidden, 6*self.hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(6 * self.hidden),
            nn.ReLU(),
            nn.Conv2d(6*self.hidden, 4*self.hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * self.hidden),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
        )
        self.classi_detect = nn.Sequential(
            nn.Linear(3136, 4*self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(4*self.hidden, 3*self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(3*self.hidden, 3*7),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.extraction(input)
        x = torch.flatten(x, 1)
        x = self.classi_detect(x)
        x = self.sigmoid(x)

        output = x.reshape(len(x), 3, 7)

        return output


class DetectionCriterion(nn.Module):
    def __init__(self):
        super(DetectionCriterion, self).__init__()
        self.criterionBCE = nn.BCELoss(reduction="sum")
        self.criterionMSE = nn.MSELoss(reduction="sum")

    def forward(self, output, target):
        targetObject = (target[:,:,0] == 1.0)*1.0
        targetnoObject = (target[:, :, 0] == 0.0) * 1.0

        targetClasse = torch.zeros((len(target), 3, 3))
        for i in range(len(target)):
            for j in range(len(target[0])):
                targetClasse[i, j, int(target[i, j, -1])] = 1

        xywhLoss = self.criterionMSE(output[:,:,1:4], target[:,:,1:4])
        classLoss = self.criterionBCE(output[:,:,4:], targetClasse)
        noObjectLoss = self.criterionBCE(output[:,:,0], targetnoObject)
        objectLoss = self.criterionBCE(output[:, :, 0], targetObject)
        criterion = 2*classLoss + 4*xywhLoss + objectLoss #+ 0.5*noObjectLoss
        return criterion

