from torch import nn

from models.utils.abstract_model_wrapper import AbstractModelWrapper
from models.utils.blocks import ConvBlock


class BITVehicleClassifierNet(AbstractModelWrapper):
    def __init__(self, input_resolution, num_classes):
        super().__init__()
        self.conv1 = ConvBlock(3, 32)
        shape = self.conv1.get_output_shape(input_resolution)
        self.conv2 = ConvBlock(32, 64)
        shape = self.conv2.get_output_shape(shape)
        n_features = 64 * shape[0] * shape[1]
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.layer1 = nn.Linear(n_features, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.lrelu(x)
        x = self.layer2(x)
        x = self.lrelu(x)
        x = self.layer3(x)
        return x
