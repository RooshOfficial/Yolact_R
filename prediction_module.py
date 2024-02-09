#proto_out : [n, 32, 138, 138]
coef_dim=proto_out.shape[1]
num_classes=81
aspect_ratios: [1, 1 / 2, 2]
class PredictionModule(nn.Module):
    def __init__(self, in_channels, coef_dim):
        super().__init__()

        self.num_classes = 81
        self.coef_dim = coef_dim
        self.num_priors = 3            # num of anchor box for each pixel of feature map

        self.upfeature = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        out_channels = 256
        self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, kernel_size=3, padding=1)
        self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.coef_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upfeature(x)
        x = self.relu(x)
        conf = self.conf_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        bbox = self.bbox_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        coef_test = self.mask_layer(x)
        print('mask layer output shape : ', coef_test.shape)
        coef = self.mask_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.coef_dim)       
        # mask_layer output shape : [n, 96, 69, 69] / In order to make it's shape [n, 69*69*3, 32], use permute and contiguous.
        print('Changed shape : ', coef.shape)
        coef = torch.tanh(coef)

        return {'box': bbox, 'class': conf, 'coef': coef}
prediction_layers = nn.ModuleList()
prediction_layers.append(PredictionModule(in_channels=256, coef_dim=coef_dim))
print(prediction_layers[0](dffn_outs[0]))

predictions = {'box': [], 'class': [], 'coef': []}
for i in range(len(dffn_outs)) :
    p=prediction_layers[0](dffn_outs[i])
    for key, value in p.items() :
        predictions[key].append(value)
print(predictions.keys())