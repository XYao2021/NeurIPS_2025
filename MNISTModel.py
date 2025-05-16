import torch
from torch import nn
import torch.nn.functional as F

class MNISTModel(nn.Module):  # Improve the model V0 with nonlinear activation function nn.Relu()
    def __init__(self, input_shape=784,
                 output_shape=10,
                 hidden_units=50):
        super().__init__()
        self.layer_1 = nn.Sequential(nn.Flatten(),  # Equal to x.view(-1, 784)
                                     nn.Linear(in_features=input_shape, out_features=hidden_units),
                                     nn.ReLU())
        self.layer_2 = nn.Linear(in_features=hidden_units, out_features=output_shape)

        @torch.no_grad()
        def initial_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                # nn.init.zeros_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # nn.init.zeros_(m.bias)

        def zero_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.zeros_(m.bias)

        def zero_bias(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.zeros_(m.bias)

        self.layer_1.apply(initial_weights)
        self.layer_2.apply(initial_weights)

    def forward(self, x):
        x = self.layer_1(x)
        return self.layer_2(x)

class KMNISTModel(nn.Module):
    def __init__(self):
        super(KMNISTModel, self).__init__()
        # Input: 1 x 28 x 28 (grayscale image)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # Output: 32 x 28 x 28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64 x 28 x 28
        self.pool = nn.MaxPool2d(2, 2)  # Reduces spatial dims by 2
        # self.dropout = nn.Dropout(0.25)
        # After two pooling operations: 28 -> 14 -> 7, so feature map: 64 x 7 x 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # KMNIST has 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)         # 32 x 14 x 14
        x = F.relu(self.conv2(x))
        x = self.pool(x)         # 64 x 7 x 7
        # x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64*7*7)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x
