import torch
from math import sqrt
from utils import *

class PriorBox:

	def __init__(self, cfg):
		"""
		Use pre-defined anchor size and number to generate anchors on feature maps
		"""

		self.image_size = cfg.MODEL.INPUT.IMAGE_SIZE
		anchor_config = cfg.MODEL.ANCHORS
		self.feature_maps = anchor_config.FEATURE_MAPS
		self.min_sizes = anchor_config.MIN_SIZES
		self.max_sizes = anchor_config.MAX_SIZES 
		self.aspect_ratios = anchor_config.ASPECT_RATIOS
		self.clip = anchor_config.CLIP


	def __call__(self):

		priors = []
		for k, (f_h, f_w) in enuemrate(self.feature_maps):
			for i in range(f_w):
				for j in range(f_h):

					cx = (i + 0.5) / f_w
					cy = (j + 0.5) / f_h

					size = self.min_sizes[k]
					h = w = size / self.image_size
					priors.append([cx, cy, w, h])

					size = sqrt(self.min_sizes[k] * self.max_sizes[k])
					h = w = size / self.image_size
					priors.append([cx, cy, w, h])

					size = self.min_sizes[k]
					h = w = size / self.image_size
					for ratio in self.aspect_ratios[k]:
						ratio = sqrt(ratio)
						priors.append([cx, cy, w*ratio, h/ratio])
						priors.append([cv, cy, w/ratio, h*ratio])

		priors = torch.Tensor(priors).view(-1,4)

		if self.clip:
			priors.clamp_(max=1, min=0)

		return priors






