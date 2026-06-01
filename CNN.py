import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Class for a Convolutional Neural Network (CNN) model for image classification.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int = 10,
    ) -> None:
        """
        Initializes the CNN model with the specified input shape and number of classes.

        Args:
            input_shape (tuple[int, int, int]): The shape of the input images (height, width, channels).
            num_classes (int): The number of classes for classification.

        Returns:
            None
        """
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[2],
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
 
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the CNN model.

        Args:
            batch (torch.Tensor): The input tensor representing a batch of images.

        Returns:
            torch.Tensor: The output tensor representing the class scores for each input image.
        """
        batch = self.pool(self.relu(self.conv1(batch)))
        batch = self.pool(self.relu(self.conv2(batch)))
        batch = batch.reshape(-1, 64 * 16 * 16)
        batch = self.relu(self.fc1(batch))
        batch = self.dropout(batch)
        batch = self.fc2(batch)

        return batch
