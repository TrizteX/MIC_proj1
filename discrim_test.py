class Discriminator(nn.Module):
      def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv1 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv4 = nn.Conv2d(512, 1, 4, 1)

        self.l_relu = nn.LeakyReLU(0.2, inplace = True)

        
        self.bn1 =  nn.BatchNorm2d(128)
        self.bn2 =  nn.BatchNorm2d(256)
        self.bn3 =  nn.BatchNorm2d(512)
        """self.features = nn.Sequential(
            #nn.Conv2d(3, 64, 4, 2, 1),
            #nn.LeakyReLU(0.2, inplace = True),
            #nn.Conv2d(64, 128, 4, 2, 1),
            #nn.BatchNorm2d(128),
            #nn.LeakyReLU(0.2, inplace = True),
            #nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4, 1)

        )"""
      def forward(self, x):
           print("Input Shape {}".format(x.shape))
           x = self.l_relu(self.conv0(x))
           print("Conv0 Shape {}".format(x.shape))
           x = self.bn1(self.l_relu(self.conv1(x)))
           print("Conv1 Shape {}".format(x.shape))
           x = self.bn2(self.l_relu(self.conv2(x)))
           print("Conv2 Shape {}".format(x.shape))
           x = self.bn3(self.l_relu(self.conv3(x)))
           print("Conv3 Shape {}".format(x.shape))
           x = self.conv4(x)
           print("Conv4 Shape {}".format(x.shape))
           print("Final Shape {}.format".format(x.view(-1).shape))
           return x.view(-1)

      def clip(self, c=0.01):
        """Weight clipping in (-c, c)"""

        for p in self.parameters():
            p.data.clamp_(-c, c)
