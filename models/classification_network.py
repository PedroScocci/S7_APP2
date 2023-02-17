import torch.nn as nn

class AlexNetClassification(nn.Module):
    def __init__(self, num_class=3):
        super(AlexNetClassification, self).__init__()
        self.hidden=4
        self.extraction = nn.Sequential(
            nn.Conv2d(1, 8*self.hidden, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(8*self.hidden, 8*self.hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(8*self.hidden, 8*self.hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*self.hidden, self.hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
        )
        self.flatten = nn.Flatten(1, -1)
        self.classification = nn.Sequential(
            nn.Linear(1936, 16*self.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(16*self.hidden, 3*self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(3 * self.hidden, 3),
            nn.Sigmoid()
        )


    def create_model(self):
        model = nn.Sequential(self.extraction, self.flatten, self.classification)
        return model

    def create_criterion(self):
        criterion = nn.BCELoss(reduction="sum")
        return criterion