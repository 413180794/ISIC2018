from torch import nn
import torch
import torchvision

from models.models import DecoderBlock, ConvRelu


class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            vgg - encoder pre-trained with VGG11
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.pool = nn.MaxPool2d(2, 2)

        if pretrained == 'vgg':
            self.encoder = torchvision.models.vgg16(pretrained=True).features
        else:
            self.encoder = torchvision.models.vgg16(pretrained=False).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)
        self.center_Conv2d = nn.Conv2d(num_filters * 8, num_classes, kernel_size=1)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))

        center_conv = self.center_Conv2d(center)
        x_out_empty_ind1 = nn.AvgPool2d(kernel_size=center_conv.size()[2:])(center_conv)
        x_out_empty_ind1 = torch.squeeze(x_out_empty_ind1)

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        x_out_mask = self.final(dec1)
        x_out_empty_ind2 = nn.MaxPool2d(kernel_size=x_out_mask.size()[2:])(x_out_mask)
        x_out_empty_ind2 = torch.squeeze(x_out_empty_ind2)
        #print(x_out_empty_ind.size()) # [8,5,1,1]

        #return x_out_mask, x_out_empty_ind1, x_out_empty_ind2
        return x_out_mask, x_out_empty_ind1, x_out_empty_ind2