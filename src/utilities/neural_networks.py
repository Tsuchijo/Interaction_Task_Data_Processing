## Class which contains neural network architectures and data loaders
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import cv2
import torch
import os

## A basic Classifier CNN architecture from chatgpt
# Assumes a square input image for processing
class Simple_CNN(nn.module):
    def __init__(self, num_classes, input_size):
        super(Simple_CNN, self).__init__()
        # Define your CNN architecture here
        # You can use Conv2d, MaxPool2d, and other layers
        # Example:
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * input_size * input_size, num_classes)

    def forward(self, x):
        # Implement the forward pass of your CNN
        # Example:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = self.fc(x)
        return x
    
## A custom dataset implementation which loads images from videos and returns them as tensors
# Loads labels from a csv file
# loads videos from a folder of mp4 files where the file name is the trial number
class VideoDataset(Dataset):
    def __init__(self, video_path, label_path, transform=None):
        # Initialize the dataset
        self.label_path = label_path
        self.transform = transform
        self.labels = pd.read_csv(label_path, index_col=0)
        # take all videos files in the video path directory and store them in a list
        self.videos = []
        for file in os.listdir(video_path):
            if file.endswith('.mp4'):
                self.videos.append(file)

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Load the video at the given index and return it as a tensor
        # Return the label as a tensor
        # Example:
        label = self.labels.iloc[idx]

        # Load the video using cv2 
        video_reader = cv2.VideoCapture(self.videos[idx])
        video_list = [video_reader.read()[1] for i in range(int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT)))]

        if self.transform:
            video_list = [self.transform(video) for video in video_list]

        # convert video list to tensor
        video_list = torch.stack(video_list)

        return video_list, label
        
        