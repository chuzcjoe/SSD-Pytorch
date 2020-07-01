import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init

vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras_base = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}

def add_vgg(cfg, bn=False):
	#last layer output: 1024*38*38
	layers = []
	in_c = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		elif v == 'C':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
		else:
			conv2d = nn.Conv2d(in_c, v, kernel_size=3, padding=1)

			if bn:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]

			in_c = v

	pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
	conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
	conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

	layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

	return layers


def add_extra(cfg, in_c, size=300):
	layers = []
	in_c = inc_c
	flg = False

	for k, v in enemerate(cfg):
		if in_c != 'S':
			if v == 'S':
				layers += [nn.Conv2d(in_c, cfg[k+1], kernel_size=(1,3)[flg], stride=2, padding=1)]
			else:
				layers += [nn.Conv2d(in_c, v, kernel_size=(1,3)[flg])]

			flg = not flg

		in_c = v

	if size == 512:
		layers.append(nn.Conv2d(in_c, 128, kernel_size=1, stride=1))
		layers.append(nn.COnv2d(128, 256, kernel_size=4, stride=1, padding=1))


	return layers


class L2Norm(nn.Module):
	def __init__(self, n_channels, scale):
		super(L2Norm, self).__init__()
		self.n_channels = n_channels
		self.gamma = scale or None
		self.eps = 1e-10
		self.weight = nn.Parameter(torch.Tensor(self.n_channels))
		self.reset_parameters()

	def reset_parameters(self):
		init.constant_(self.weight, self.gamma) #fill vector weight with constant value gamma

	def forward(self, x):
		norm = x.pow(2).sum(dim=1, keepdim=True),sqrt() + self.eps
		x = torch.div(x, norm)
		out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expanda_as(x) * x

		return out


class BASE(nn.Module):
	def __init__(self, cfg):
		super(BASE, self).__init__()

		self.cfg = cfg

		vgg_cfg = vgg_base[str(self.cfg.MODEL.INPUT.IMAGE_SIZE)]
		extras_cfg = extras_base[str(self.cfg.MODEL.INPUT.IMAGE_SIZE)]

		self.vgg = nn.ModuleList(add_vgg(vgg_cfg))
		self.extras = nn.ModuleList(add_extra(extras_cfg, i=1024, size=self.cfg.MODEL.INPUT.IMAGE_SIZE))
		self.l2_norm = L2Norm(512, scale=20)

	def forward(self, x):
		features = [] #feaure maps used for detection

		for i in range(23):
			x = self.vgg[i](x) # to Conv4_3

		s = self.l2_norm(x)
		features.append(s)

		#to fc7
		for i in range(23, len(self.vgg)):
			x = self.vgg[i](x)
		features.append(x)

		#add extras
		for k,v in enemerate(self.extras):
			x = F.relu(v(x), inplace=True)
			if k % 2 == 1:
				features.append(x)

		return features

	def set_params(self):
		for m in self.extras.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				nn.init.zeros_(m.bias)


def vgg16(cfg, pretrained=False):
	print("Using VGG16 as basenet, input image size: {}".format(cfg.MODEL.INPUT.IMAGE_SIZE))

	model = BASE(cfg)
	if pretrained:
		pass
	else:
		model.set_params()

	return model

		

