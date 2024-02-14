"""
    MIT License
    Copyright (c) 2024 Reza NasiriGerdeh

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import torch
from torch import nn, Tensor


class CNN(nn.Module):
    """ A simple convolutional network for testing purposes """
    def __init__(self, num_classes: int) -> None:
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)))
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x_t: Tensor) -> Tensor:
        out = self.block1(x_t)
        out = self.block2(out)
        out = self.avg_pool(out)
        out = torch.flatten(input=out, start_dim=1, end_dim=-1)
        out = self.classifier(out)
        return out
