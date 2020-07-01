import torch
import math

def decode(locs, priors, center_var, size_var):
	"""
	convert regression results to bboxes information
	"""

	if priors.dim() + 1 == locs.dim():
		priors = priors.unsqueeze(0)

	return torch.cat([locs[...,:2] * center_var * priors[..., 2:] + priors[..., :2], 
					torch.exp(locs[..., 2:] * size_var) * priors[..., 2:]], dim=locs.dim()-1)



def coord_convert(locs):
	"""
	convert [x,y,w,h] to [xmin,ymin,xmax,ymax] form
	"""

	return torch.cat([locs[..., :2]-locs[..., 2:]/2,
		locs[..., :2] + locs[..., 2:]/2], locs.dim()-1
		)