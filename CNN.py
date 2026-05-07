import numpy as np
from torch import nn

class CNN(nn.Module):
    """
    Class for a Convolutional Neural Network (CNN) model for image classification.
    """
    def __init__(self, num_classes: int = 10) -> None:
        """
        Initializes the CNN model with the specified input shape and number of classes.

        Args:
            input_shape (tuple[int, int]): The shape of the input images (height, width).
            num_classes (int): The number of classes for classification.

        Returns:
            None
        """
        super().__init__()

        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)



    def forward():
        pass
