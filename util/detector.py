import torch
from torch import nn


class detector(nn.Module):
	"""
	object cls and bbox regression based on extracted feature maps
	"""

	def __init__(self, cfg):
		super(detector, self).__init__()

		self.cfg = cfg
		self.cls_headers = nn.ModuleList()
		self.reg_headers = nn.ModuleList()

		for bboxes_per_location, out_channels in zip(cfg.MODEL.ANCHORS.BOXES_PER_LOCATION, cfg.MODEL.ANCHORS.OUT_CHANNELS):
			self.cls_headers.append(self.cls_block(out_channels, bboxes_per_location))
			self.reg_headers.append(self.reg_block(out_channels, bboxes_per_location))

		self.set_params()

	def cls_block(self, out, bboxes):
		return nn.Conv2d(out, bboxes * self.cfg.DATA.DATASET.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

	def reg_block(self, out, bboxes):
		return nn.Conv2d(out, bboxes * 4, kernel_size=3, stride=1, padding=1)

	def set_parmas(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				nn.init.zeros_(m.bias)


	def forward(self, features):
		"""
		6 feature maps from baseet go through cls and reg headers
		"""

		cls_logits = []
		bbox_preds = []

		for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
			cls_logits.append(cls_header(feature).permute(0, 2, 3, 1).contiguous()) #apply continuous for view operation use
			bbox_preds.append(reg_header(feature).permute(0, 2, 3, 1).continuous())

		batch = features[0].size(0)
		#shape: [batch, anchors, classes]
		cls_logits = torch.cat([c.view(c.size(0),-1) for c in cls_logits], dim=1).view(batch_size, -1, self.cfg.DATA.DATASET.NUM_CLASSES)

		#shape: [batch, anchors, 4]
		bbox_preds = torch.cat([b.view(b.size(0), -1) for b in bbox_preds], dim=1).view(batch_size, -1, 4)

		return cls_logits, bbox_preds