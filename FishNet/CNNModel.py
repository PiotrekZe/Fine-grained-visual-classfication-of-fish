import torch.nn as nn
import CBAM


class FishNet(nn.Module):
    def __init__(self, attention_map=False):
        super(FishNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2)
        )

        self.attention1 = CBAM.ChannelAttention(64)
        self.attention2 = CBAM.SpatialAttention()

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2)
        )

        self.attention3 = CBAM.ChannelAttention(128)
        self.attention4 = CBAM.SpatialAttention()

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout2d(0.2)
        )

        self.attention5 = CBAM.ChannelAttention(256)
        self.attention6 = CBAM.SpatialAttention()

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2)
        )

        self.attention7 = CBAM.ChannelAttention(128)
        self.attention8 = CBAM.SpatialAttention()

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2)
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1 * 1 * 64, 1024),
            nn.GELU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 570)
        )

        # 1x1 convolutional layers to get proper size for skip connections
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1)

        self.attention_map = attention_map

    def forward(self, x):
        # First layer
        x = self.layer1(x)
        x = self.attention1(x) * x
        first_layer = self.attention2(x) * x

        # Second layer
        x = self.layer2(first_layer)
        x = self.attention3(x) * x
        x = self.attention4(x) * x

        reshaped_first_layer = self.conv1(first_layer)
        reshaped_first_layer = nn.functional.interpolate(reshaped_first_layer, size=(x.size(2), x.size(3)),
                                                         mode='bilinear', align_corners=False)
        x = nn.functional.gelu(x + reshaped_first_layer)

        # Third layer
        x = self.layer3(x)
        x = self.attention5(x) * x
        x = self.attention6(x) * x

        reshaped_first_layer = self.conv2(first_layer)
        reshaped_first_layer = nn.functional.interpolate(reshaped_first_layer, size=(x.size(2), x.size(3)),
                                                         mode='bilinear', align_corners=False)
        x = nn.functional.gelu(x + reshaped_first_layer)

        # Fourth layer
        x = self.layer4(x)
        x = self.attention7(x) * x
        x = self.attention8(x) * x

        reshaped_first_layer = self.conv1(first_layer)
        reshaped_first_layer = nn.functional.interpolate(reshaped_first_layer, size=(x.size(2), x.size(3)),
                                                         mode='bilinear', align_corners=False)
        x = nn.functional.gelu(x + reshaped_first_layer)

        # Fifth layer
        x = self.layer5(x)
        reshaped_first_layer = nn.functional.interpolate(first_layer, size=(x.size(2), x.size(3)), mode='bilinear',
                                                         align_corners=False)
        x = nn.functional.gelu(x + reshaped_first_layer)

        # Dense layers
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)

        if self.attention_map:
            return nn.functional.log_softmax(x, dim=1), first_layer
        else:
            return nn.functional.log_softmax(x, dim=1)
