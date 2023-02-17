import torch
import torch.nn as nn
import numpy as np
import torchvision

class AlexNetDectection(nn.Module):
    def __init__(self):
        super(AlexNetDectection, self).__init__()
        self.hidden = 16
        self.conv1 = nn.Conv2d(1, 2*self.hidden, kernel_size=5, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(2*self.hidden)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(2*self.hidden, 6*self.hidden, kernel_size=5, padding=1, stride=2)
        self.batchNorm2 = nn.BatchNorm2d(6 * self.hidden)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(6*self.hidden, 6*self.hidden, kernel_size=3, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(6 * self.hidden)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(6*self.hidden, 6*self.hidden, kernel_size=3, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(6 * self.hidden)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(6*self.hidden, 2*self.hidden, kernel_size=3, padding=1)
        self.batchNorm5 = nn.BatchNorm2d(2 * self.hidden)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.fc1 = nn.Linear(1152, 3*self.hidden)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(3*self.hidden, 3*self.hidden)
        self. relu7 = nn.ReLU()
        self.fc3 = nn.Linear(3*self.hidden, 3*7)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.conv1(input)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.batchNorm5(x)
        x = self.relu5(x)
        x = self.maxpool5(5)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.fc3(x)
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

