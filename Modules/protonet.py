############### Proto Network ###############
'''
Use N3 which is deepest feature map and has highest resolution
'''
mask_proto_net = [(256, 3, {'padding': 1}), (256, 3, {'padding': 1}), (256, 3, {'padding': 1}),
                  (None, -2, {}), (256, 3, {'padding': 1}), (32, 1, {})]

class Protonet(nn.Module) :
    def __init__(self, mask_proto_net) :
        super().__init__()

        self.inplanes=256
        self.mask_proto_net = mask_proto_net
        self.conv1 = nn.Conv2d(self.inplanes, mask_proto_net[0][0], kernel_size=mask_proto_net[0][1], **mask_proto_net[0][2])
        self.conv2 = nn.Conv2d(self.inplanes, mask_proto_net[1][0], kernel_size=mask_proto_net[1][1], **mask_proto_net[1][2])
        self.conv3 = nn.Conv2d(self.inplanes, mask_proto_net[2][0], kernel_size=mask_proto_net[2][1], **mask_proto_net[2][2])
        self.conv4 = nn.Conv2d(self.inplanes, mask_proto_net[4][0], kernel_size=mask_proto_net[4][1], **mask_proto_net[4][2])
        self.conv5 = nn.Conv2d(self.inplanes, mask_proto_net[5][0], kernel_size=mask_proto_net[5][1], **mask_proto_net[5][2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = F.interpolate(out, scale_factor = -self.mask_proto_net[3][1], mode='bilinear', align_corners=False, **self.mask_proto_net[3][2])
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        
        
        return out

proto_out=Protonet(mask_proto_net)(n3)
print(proto_out)
print('-'*50)
print('Proto net output shape : ', proto_out.shape)