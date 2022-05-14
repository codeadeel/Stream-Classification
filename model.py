#!/usr/bin/env python3

"""
CONVNEXT-TINY MODEL
===================

The following program is the model definition for Stream Classification
"""

# %%
# Importing Libraries
import os
import pickle
from PIL import Image
import numpy as np
import torch
import torchvision as tv

# %%
# Training Transforms
training_transforms = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.AutoAugment(tv.transforms.AutoAugmentPolicy.IMAGENET),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
# Inference Transforms
inference_transforms = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
# Input Shape
dummy_input_shape = (1,3,224,224)

# %%
# Main Model Definition
class Model(torch.nn.Module):
    def __init__(self, num_classes):
        """
        This method is used to initialize Model

        Method Input
        =============
        num_classes : Number of classes to make classification model for

        Method Output
        ==============
        None
        """
        super(Model, self).__init__()
        self.convnext = tv.models.convnext_tiny(pretrained=False, progress = False, num_classes = num_classes)
        self.convnext.requires_grad_(True)
    
    def forward(self, x):
        """
        This method is used to perform forward propagation on input data

        Method Input
        =============
        x : Input data as image batch ( Batch x Channel x Height x Width)

        Method Output
        ==============
        Output results after forward propagation
        """
        return self.convnext(x)
    