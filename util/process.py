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

	def __call__(self, cls_logits, bbox_preds):

		priors = PriorBox(self.cfg)().cuda(self.cfg.DEVICE.GPU)
		batch_socres = F.softmax(cls_logits, dim=2)
		boxes = decode(bbox_preds, priors, self.cfg.MODEL.ANCHORS.CENTER_VARIANCE, self.cfg.MODEL.ANCHORS.CENTER_VARIANCE)

		#to [xmin,ymin,xmax,ymax]
		batches_boxes = coord_convert(boxes)
		batch_size = batch_socres.size(0)
		res = []

		for batch_id in range(batch_size):
			processed_boxes = []
			processed_scores = []
			processed_labels = []

			#[N, cls], [N, 4]
			per_img_scores, per_img_boxes = batch_socres[batch_id], batches_boxes[batch_id]

			for cls_id in range(1, per_img_scores.size(1)): #skip background
				socres = per_img_scores[:, cls_id]
				mask = scores > self.cfg.MODEL.TEST.CONFIDENCE_THRESHOLD
				scores = scores[mask]

				#this class has low confidence for all bboxes
				if scores.size(0) == 0:
					continue

				boxes = per_img_boxes[mask, :]

				#convert to real size
				boxes[:,0::2] *= self.width
				boxes[:,1::2] *= self.height

				#apply nms
				keep = apply_nms(boxes, scores, self.cfg.MODEL.TEST.NMS_THRESHOLD, self.cfg.MODEL.TEST.MAX_PER_CLASS)

				nmsed_boxes = boxes[keep, :]
				nmsed_labels = torch.Tensor([cls_id]*keep.size(0))
				nmsed_socres = scores[keep]

				processed_boxes.append(nmsed_boxes)
				processed_labels.append(nmsed_labels)
				processed_scores.append(nmsed_socres)


			if len(processed_boxes) == 0:
				processed_boxes = torch.empty(0, 4)
				processed_labels = torch.empty(0)
				processed_scores = torch.empty(0)

			else:
				processed_boxes = torch.cat(processed_boxes, dim=0)
				processed_labels = torch.cat(processed_labels, dim=0)
				processed_scores = torch.cat(processed_scores, dim=0)

			if processed_boxes.size(0) > self.cfg.MODEL.TEST.MAX_PER_IMAGE > 0:
				processed_scores, keep = torch.topk(processed_scores, k=self.cfg.MAX_PER_IMAGE)
				processed_boxes = processed_boxes[keep, :]
				processed_labels = processed_labels[keep]


			res.append([processed_boxes, processed_labels, processed_scores])

		return res

