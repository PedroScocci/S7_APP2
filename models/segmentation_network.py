import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(UNet, self).__init__()
        self.hidden = 25

        # Down 1
        self.conv_1_1 = nn.Conv2d(input_channels, self.hidden, (3,3), (1,), 1)
        self.relu_1_1 = nn.ReLU()
        self.conv_1_2 = nn.Conv2d(self.hidden, self.hidden, (3,3), (1,), 1)
        self.relu_1_2 = nn.ReLU()

        # Down 2
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, padding=0, stride=(2,))
        self.conv_2_1 = nn.Conv2d(self.hidden, self.hidden*2, (3,3), (1,), 1)
        #self.conv_2_1 = nn.Conv2d(input_channels, self.hidden * 2, (3, 3), (1,), 1)
        self.relu_2_1 = nn.ReLU()
        self.conv_2_2 = nn.Conv2d(self.hidden*2, self.hidden*2, (3,3), (1,), 1)
        self.relu_2_2 = nn.ReLU()

        # Down 3
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, padding=0, stride=(2,))
        self.conv_3_1 = nn.Conv2d(self.hidden*2, self.hidden*4, (3,3), (1,), 1)
        self.relu_3_1 = nn.ReLU()
        self.conv_3_2 = nn.Conv2d(self.hidden*4, self.hidden*4, (3,3), (1,), 1)
        self.relu_3_2 = nn.ReLU()

        # Down 4
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, padding=0, stride=(2,))
        self.conv_4_1 = nn.Conv2d(self.hidden*4, self.hidden*8, (3,3), (1,), 1)
        self.relu_4_1 = nn.ReLU()
        #self.conv_4_2 = nn.Conv2d(self.hidden*8, self.hidden*8, (3,3), (1,), 1)
        self.conv_4_2 = nn.Conv2d(self.hidden*8, self.hidden*4, (3,3), (1,), 1)
        self.relu_4_2 = nn.ReLU()

        # Down 5
        #self.maxpool_5 = nn.MaxPool2d(kernel_size=2, padding=0, stride=(2,))
        #self.conv_5_1 = nn.Conv2d(self.hidden*8, self.hidden*16, (3,3), (1,), 1)
        #self.relu_5_1 = nn.ReLU()
        #self.conv_5_2 = nn.Conv2d(self.hidden*16, self.hidden*8, (3,3), (1,), 1)
        #self.relu_5_2 = nn.ReLU()

        # Up 6
        #self.upsample_6 = nn.ConvTranspose2d(self.hidden*8, self.hidden*8, (2, 2), (2,), (0,))
        #self.conv_6_1 = nn.Conv2d(self.hidden*16, self.hidden*8, (3,3), (1,), 1)
        #self.relu_6_1 = nn.ReLU()
        #self.conv_6_2 = nn.Conv2d(self.hidden*8, self.hidden*4, (3,3), (1,), 1)
        #self.relu_6_2 = nn.ReLU()

        # Up 7
        self.upsample_7 = nn.ConvTranspose2d(self.hidden*4, self.hidden*4, (3, 3), (2,), (0,))
        self.conv_7_1 = nn.Conv2d(self.hidden*8, self.hidden*4, (3,3), (1,), 1)
        self.relu_7_1 = nn.ReLU()
        self.conv_7_2 = nn.Conv2d(self.hidden*4, self.hidden*2, (3,3), (1,), 1)
        self.relu_7_2 = nn.ReLU()

        # Up 8
        self.upsample_8 = nn.ConvTranspose2d(self.hidden*2, self.hidden*2, (2, 2), (2,), (0,))
        self.conv_8_1 = nn.Conv2d(self.hidden*4, self.hidden*2, (3,3), (1,), 1)
        self.relu_8_1 = nn.ReLU()
        self.conv_8_2 = nn.Conv2d(self.hidden*2, self.hidden*1, (3,3), (1,), 1)
        self.relu_8_2 = nn.ReLU()

        # Up 9
        self.upsample_9 = nn.ConvTranspose2d(self.hidden*1, self.hidden*1, (3, 3), (2,), (0,))
        self.conv_9_1 = nn.Conv2d(self.hidden*2, self.hidden*1, (3,3), (1,), 1)
        self.relu_9_1 = nn.ReLU()
        self.conv_9_2 = nn.Conv2d(self.hidden*1, self.hidden*1, (3,3), (1,), 1)
        self.relu_9_2 = nn.ReLU()

        self.output_conv = nn.Conv2d(self.hidden, n_classes, kernel_size=1)

    def forward(self, x):
        # Down 1
        down1 = self.relu_1_2(self.conv_1_2(self.relu_1_1(self.conv_1_1(x))))

        # Down 2
        down2 = self.relu_2_2(self.conv_2_2(self.relu_2_1(self.conv_2_1(self.maxpool_2(down1)))))
        #down2 = self.relu_2_2(self.conv_2_2(self.relu_2_1(self.conv_2_1(x))))

        # Down 3
        down3 = self.relu_3_2(self.conv_3_2(self.relu_3_1(self.conv_3_1(self.maxpool_3(down2)))))

        # Down 4
        down4 = self.relu_4_2(self.conv_4_2(self.relu_4_1(self.conv_4_1(self.maxpool_4(down3)))))

        # Down 5
        #down5 = self.relu_5_2(self.conv_5_2(self.relu_5_1(self.conv_5_1(self.maxpool_5(down4)))))

        # Up 6
        #concat6 = torch.concat((self.upsample_6(down5), down4), 1)
        #up6 = self.relu_6_2(self.conv_6_2(self.relu_6_1(self.conv_6_1(concat6))))

        # Up 7
        #temp_upsample7 = self.upsample_7(up6)
        #concat7 = torch.concat((self.upsample_7(up6), down3), 1)
        concat7 = torch.concat((self.upsample_7(down4), down3), 1)
        up7 = self.relu_7_2(self.conv_7_2(self.relu_7_1(self.conv_7_1(concat7))))

        # Up 8
        concat8 = torch.concat((self.upsample_8(up7), down2), 1)
        up8 = self.relu_8_2(self.conv_8_2(self.relu_8_1(self.conv_8_1(concat8))))

        # Up 9
        concat9 = torch.concat((self.upsample_9(up8), down1), 1)
        up9 = self.relu_9_2(self.conv_9_2(self.relu_9_1(self.conv_9_1(concat9))))

        # Out
        out = self.output_conv(up9)
        #out = self.output_conv(up8)

        return out
