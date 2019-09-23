import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Model(nn.Module):
    
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.BN_affine = False
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64, affine=self.BN_affine)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128, affine=self.BN_affine)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256, affine=self.BN_affine)
        
        self.fc1 = nn.Linear(256*143, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)
        self._weight_init()
        
    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(-1, 256*143)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        
        return x

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_((m.weight))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if not self.BN_affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0)


class CNN_Model_withDropout(nn.Module):

    def __init__(self):
        super(CNN_Model_withDropout, self).__init__()
        self.BN_affine = False
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64, affine=self.BN_affine)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128, affine=self.BN_affine)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256, affine=self.BN_affine)

        self.fc1 = nn.Linear(256 * 143, 1024)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 2)
        self._weight_init()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(-1, 256 * 143)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        x = F.softmax(self.fc3(x), dim=1)

        return x

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_((m.weight))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if not self.BN_affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0)