import torch
import torch.nn as nn

class DFFN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1x1 convolution for P5
        self.p5_conv = nn.Conv2d(2048, 256, kernel_size=1)

        # Upsampling
        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 1x1 convolutions for C4 and C3
        self.c4_conv = nn.Conv2d(1024, 256, kernel_size=1)
        self.c3_conv = nn.Conv2d(512, 256, kernel_size=1)

        # 3x3 convolutions for N3, N4, N5
        self.n3_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.n4_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.n5_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # The size of torch.cat([p4, m3], dim=1) is torch.Size([1, 512, 35, 35]).

        # This is because:

        # p4 has a size of torch.Size([1, 256, 35, 35]).
        # m3 has a size of torch.Size([1, 256, 35, 35]).
        # When you concatenate two tensors along dimension 1, the number of channels is added together.
        # Therefore, the resulting tensor has 512 channels, the same number of channels as p4 and m3 combined. The other dimensions (height and width) remain the same.

        # Downsampling layers
        self.m3_down = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.m4_down = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.n6_down = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.n7_down = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

    def forward(self, c3, c4, c5):
        # Calculate P5
        p5 = self.p5_conv(c5)

        # Calculate P4
        p4_upsample = self.upsample2x(p5)
        # p4_upsample = F.interpolate(p5, size=target_size, mode='bilinear', align_corners=False)
        p4_conv = self.c4_conv(c4)
        p4 = p4_upsample + p4_conv

        # Calculate P3
        p3_upsample = self.upsample2x(p4)
        p3_conv = self.c3_conv(c3)
        p3 = p3_upsample + p3_conv

        # Calculate N3
        n3 = self.n3_conv(p3)

        # Calculate M3 and N4
        m3 = self.m3_down(n3)
        n4 = self.n4_conv(torch.cat([p4, m3], dim=1))

        # Calculate M4 and N5
        m4 = self.m4_down(n4)
        n5 = self.n5_conv(torch.cat([p5, m4], dim=1))

        # Calculate N6 and N7
        n6 = self.n6_down(n5)
        n7 = self.n7_down(n6)

        return n3, n4, n5, n6, n7

# Example usage
# c3 = torch.randn(1, 512, 69, 69)
# c4 = torch.randn(1, 1024, 35, 35)
# c5 = torch.randn(1, 2048, 18, 18)
    
c3 = backbone[1]
c4 = backbone[2]
c5 = backbone[3]

dfnn = DFFN()
# n3, n4, n5, n6, n7 = dfnn(c3, c4, c5)
dffn_outs = dfnn(c3, c4, c5)


# Print output sizes
print("N3 size:", dffn_outs[0].shape)
print("N4 size:", dffn_outs[1].shape)
print("N5 size:", dffn_outs[2].shape)
print("N6 size:", dffn_outs[3].shape)
print("N7 size:", dffn_outs[4].shape)

# fused_features = n3, n4, n5, n6, n7