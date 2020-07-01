import torch.nn as nn
import torch.nn.functional as F
import torch

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
		elif v = 'C':
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
		

