import torch.nn as nn
import torch
import torch.nn.functional as F
import math

def hard_negative_mine(bg, labels, neg_pos_ratio=3):
	"""
	bg: (N, num_priors) the confidence level of background
	labels: (N, num_priors)
	"""

	pos_mask = labels > 0
	num_pos = pos_mask.long().sum(dim=1, keepdim=True)
	num_neg = num_pos * neg_pos_ratio

	bg[pos_mask] = -math.inf

	#sort
	_, idxes = bg.sort(dim=1, descending=True) #idxes shape: (N, num_priors)
	_, orders = idxes.sort(dim=1)

	neg_mask = orders < num_neg

	return pos_mask | neg_mask


class loss(nn.Module):
	def __init__(self, neg_pos_ratio):
		"""
		cls: cross entropy
		reg: smooth_l1_loss
		"""
		super(loss, self).__init__()
		self.neg_pos_ratio = neg_pos_ratio

	def forward(self, conf, pred_locs, gt_labels, gt_locs):
		"""
		conf: (batch, num_priors, classes)
		pred_locs: (batch, num_priors, 4)
		labels: (batch, num_priors)  -> gt labels
		gt_locs: (batch, num_priors, 4)
		"""

		classes = conf.size(2)

		with torch.no_grad():
			n_logsoftmax = -F.log_softmax(conf, dim=2)[:, :, 0]
			mask = hard_negative_mine(n_logsoftmax, gt_labels, self.neg_pos_ratio)

		conf = conf[mask, :]
		cls_loss = F.cross_entropy(conf.view(-1, classes), gt_labels[mask], reduction='sum')

		pos_mask = gt_labels > 0
		pred_locs = pred_locs[pos_mask, :].view(-1, 4)
		gt_locs = gt_locs[pos_mask, :].view(-1, 4)
		reg_loss = F.smooth_l1_loss(pred_locs, gt_locs, reduction='sum')

		num_pos = gt.loss.size(0)
		return reg_loss / num_pos, cls_loss / num_pos
