import torch
from torch import nn
from model.net import vgg16
from util.detector import *
from util.prior import *
from util.utils import *
from util.process import *
import numpy as np
import time


class SSD(nn.Module):
	def __init__(self, cfg):
		super(SSD, self).__init__()

		self.cfg = cfg

		self.backbone = vgg16(cfg, pretrained=False)

		#cls and reg
		self.detector = detector(cfg)

		#nms
		self.process = Process(cfg)

		#prior anchors
		self.prior = PriorBox(self.cfg)()

	def forward(self, x):
		features = self.backbone(x)

		cls_logits, bbox_preds = self.detector(features)

		return cls_logits, bbox_preds

	def forward_with_nms(self, x):
		cls_logits, bbox_preds = self.forward(x)
		detections = self.process(cls_logits, bbox_preds)

		return detections