import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    '''
    Convolutional neural network for a 28x28 grayscale image.
    '''
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1)
        self.fc1 = nn.Linear(5 * 5 * 16, 80)
        self.fc2 = nn.Linear(80, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 5 * 5 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Softmax layer is not required since it is used in the Loss function
        return x


class ProductionCNN(CNN):
    '''
        Fully Convoluted version of CNN to be able to apply to any image size.
        It can be instantiated by passing a CNN's state_dict with pretrained models.
    '''

    def __init__(self, state_dict=None):
        super(ProductionCNN, self).__init__()
        if state_dict is not None:
            self.load_state_dict(state_dict)

            # This stride determines the stride of the 28*28 convolutional sliding window
            self.conv1.stride = 1

            # Convert fully connected layer to their convolutional equivalents
            inch = 16
            out = 80
            kernel = 5

            self.last_conv1 = nn.Conv2d(in_channels=inch, out_channels=out, kernel_size=kernel)
            self.last_conv1.weight.data.copy_(self.fc1.weight.data.view(out, inch, kernel, kernel))
            self.last_conv1.bias.data.copy_(self.fc1.bias.data)

            self.last_conv2 = nn.Conv2d(in_channels=self.fc2.in_features, out_channels=10, kernel_size=1)
            self.last_conv2.weight.data.copy_(self.fc2.weight.data.view(*self.fc2.weight.data.shape, 1, 1))
            self.last_conv2.bias.data.copy_(self.fc2.bias.data)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.last_conv1(x))
        x = self.last_conv2(x)
        return x


class CNN2(nn.Module):
    '''
    Convolutional neural network for 28x28 grayscale image.
    '''
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 32, 7 * 7 * 32 // 2)
        self.fc2 = nn.Linear(7 * 7 * 32 // 2, 7 * 7 * 32 // 4)
        self.fc3 = nn.Linear(7 * 7 * 32 // 4, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # We don't need the softmax layer since it is used in the Loss function
        return x


class ProductionCNN2(CNN2):
    '''
        Fully Convoluted version of CNN2 to be able to apply to any image size.
        It can be instantiated by passing a CNN2's state_dict with pretrained models.
    '''

    def __init__(self, state_dict=None):
        super(ProductionCNN2, self).__init__()
        if state_dict is not None:
            self.load_state_dict(state_dict)

            # This stride determines the stride of the 28*28 convolutional sliding window
            self.conv1.stride = 1

            # Convert fully connected layer to their convolutional equivalents.
            inch = 32
            out = 7 * 7 * 32 // 2
            kernel = 7

            self.last_conv1 = nn.Conv2d(in_channels=inch, out_channels=out, kernel_size=kernel)
            self.last_conv1.weight.data.copy_(self.fc1.weight.data.view(out, inch, kernel, kernel))
            self.last_conv1.bias.data.copy_(self.fc1.bias.data)

            self.last_conv2 = nn.Conv2d(in_channels=self.fc2.in_features, out_channels=self.fc2.out_features,
                                        kernel_size=1)
            self.last_conv2.weight.data.copy_(self.fc2.weight.data.view(*self.fc2.weight.data.shape, 1, 1))
            self.last_conv2.bias.data.copy_(self.fc2.bias.data)

            self.last_conv3 = nn.Conv2d(in_channels=self.fc3.in_features, out_channels=10, kernel_size=1)
            self.last_conv3.weight.data.copy_(self.fc3.weight.data.view(*self.fc3.weight.data.shape, 1, 1))
            self.last_conv3.bias.data.copy_(self.fc3.bias.data)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Here we apply the convolutions corresponding to CNN2's fully connected layers
        x = F.relu(self.last_conv1(x))
        x = F.relu(self.last_conv2(x))
        x = self.last_conv3(x)
        # We don't need the softmax layer since it is used in the Loss function
        return x
