import torch
import torch.nn as nn
import numpy as np
import torchvision

class AlexNetDectection(nn.Module):
    def __init__(self):
        super(AlexNetDectection, self).__init__()
        self.hidden = 16
        self.extraction = nn.Sequential(
            nn.Conv2d(1, 2*self.hidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(2*self.hidden, 4*self.hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(4*self.hidden, 4*self.hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*self.hidden, 2*self.hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.hidden, self.hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
        )
        self.classi_detect = nn.Sequential(
            nn.Linear(8464, 2*self.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2*self.hidden, 2*self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.hidden, 3*7),
        )
        self.softmax = nn.Softmax()
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
        self.criterionBCE = nn.BCELoss()
        self.criterionMSE = nn.MSELoss()

    def forward(self, output, target):
        targetObject = (target[:,:,0] == 1.0)*1.0
        targetNoObject = (target[:,:,0] == 0.0)*1.0

        targetClasse = torch.zeros((len(target), 3, 3))
        for i in range(len(target)):
            for j in range(len(target[0])):
                targetClasse[i, j, int(target[i, j, -1])] = 1

        xywhLoss = self.criterionMSE(output[:,:,3:6], target[:,:,1:4])
        classLoss = self.criterionBCE(output[:,:,:3], targetClasse)
        objectLoss = self.criterionBCE(output[:,:,0], targetObject)
        noObjectLoss = self.criterionBCE(output[:,:,0], targetNoObject)
        criterion = 2*classLoss + 5*xywhLoss + 0.5*noObjectLoss + objectLoss
        return criterion

