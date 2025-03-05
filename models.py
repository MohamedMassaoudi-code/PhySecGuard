"""
models.py
---------
This module contains the definition of the neural network models used in PhySecGuard.
Currently, it provides a SimpleCNN model for adversarial vulnerability evaluation on image datasets like MNIST.
"""

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for image classification.
    
    This model is designed primarily for datasets like MNIST (28x28 grayscale images).
    It includes:
      - Two convolutional layers
      - Two max pooling layers
      - Two fully connected layers
      
    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        pool (nn.MaxPool2d): Max pooling layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        relu (nn.ReLU): ReLU activation function.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # First convolution: 1 input channel, 32 output channels, 3x3 kernel, same padding
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Second convolution: 32 input channels, 64 output channels, 3x3 kernel, same padding
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Max pooling layer with kernel size 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers; assuming input images are 28x28 (e.g., MNIST)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply first convolution, followed by ReLU activation and pooling
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        # Apply second convolution, followed by ReLU activation and pooling
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        # Apply the first fully connected layer with ReLU activation
        x = self.relu(self.fc1(x))
        # Apply the second fully connected layer (output layer)
        x = self.fc2(x)
        return x
