import torch
import torchvision
from prior import PriorBox
import torch.nn.functional as F
from utils import *


def apply_nms(boxes, socres, thresh, max_count=-1):
	"""
	boxes: 'xyxy' mode
	"""

	keep = torchvision.ops.nms(boxes, socres, thresh)
	if max_count > 0:
		keep = keep[:max_count]

	return keep

class Process:
	def __init__(self, cfg):
		self.cfg = cfg
		self.width = cfg.MODEL.INPUT.IMAGE_SIZE
		self.height = cfg.MODEL.INPUT.IMAGE_SIZE

	def __call__(self, cls_logits, bbox_pred):

		priors = PriorBox(self.cfg)().cuda(self.cfg.DEVICE.GPU)
		batch_socres = F.softmax(cls_logits, dim=2)
