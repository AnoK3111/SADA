# coding: utf-8
import math
import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F




class Local_Alignment_Loss(nn.Module):
	def __init__(self, hparams, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.hparams = hparams
		self.channel = 512 if self.hparams["resnet18"] else 2048
		self.hidden_layer_size = 2048
		self.device = self.hparams["device"]
		# self.local_projection = nn.Sequential(
		#     nn.Conv2d(self.channel, self.hparams["local_projection_size"], kernel_size=1),
		# )
		self.local_projection = nn.Sequential(
		    nn.Conv2d(self.channel, self.hidden_layer_size, kernel_size=1),
			nn.ReLU(),
			nn.Conv2d(self.hidden_layer_size, self.hparams["local_projection_size"], kernel_size=1),
		)
		self.mse = nn.MSELoss()

	def forward(self, x):
		x = x.to(self.device)
		x = self.local_projection(x)
		x = F.normalize(x, dim=1)
		contrast_list = []
		index = torch.arange(x.shape[0]).to(x.device)
		for env in range(self.hparams['k']):
			features = x[index % self.hparams['k'] == env]
			contrast_list.append(features)
		loss = 0
		for i in range(1, self.hparams['k']):
			loss += self.mse(contrast_list[0], contrast_list[i])
		return loss / (self.hparams['k'] - 1)
	

class Supervised_Contrast_Loss(nn.Module):
	def __init__(self, hparams, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.hparams = hparams

	def forward(self, x, y,eps=1e-6):
		loss = 0
		features = F.normalize(x, dim=1)
		batch_size = features.shape[0]
		y = y.contiguous().view(-1, 1)
		label_mask = torch.eq(y, y.T).float()
		# compute logits
		anchor_dot_contrast = torch.div(
			torch.matmul(features, features.T),
			self.hparams['temperature'])
		# for numerical stability
		logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
		logits = anchor_dot_contrast - logits_max.detach()

		# mask-out self-contrast cases
		logits_mask = torch.scatter(
			torch.ones_like(label_mask),
			1,
			torch.arange(batch_size).view(-1, 1).to(y.device),
			0
		)
		label_mask = label_mask * logits_mask   

		# compute log_prob
		exp_logits = torch.exp(logits) * logits_mask
		log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

		# compute mean of log-likelihood over positive
		mean_log_prob_pos = (label_mask * log_prob).sum(1) / (label_mask.sum(1) + eps)

		# loss
		loss = - mean_log_prob_pos
		loss = loss.mean()
		return loss
	

class Domain_Proxy_Contrast_Loss(nn.Module):
	def __init__(self, hparams, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.hparams = hparams

	def forward(self, x, y,eps=1e-6):
		loss = 0
		batch_size = x.shape[0]
		x = F.normalize(x, dim=1)
		contrastive_features = x

		# contrastive_features = x.reshape(batch_size//self.hparams["k"], self.hparams["k"], -1)[:,0]
		proxy_features = x.reshape(batch_size//self.hparams["k"], self.hparams["k"], -1).mean(1)
		#contrastive_features = F.normalize(contrastive_features, dim=1)
		#proxy_features = F.normalize(proxy_features, dim=1)
        
		proxy_y = y.reshape(batch_size//self.hparams["k"], self.hparams["k"])[:,0]
		y = y.contiguous().view(-1, 1)
		proxy_y = proxy_y.contiguous().view(-1, 1)
		label_mask = torch.eq(proxy_y, y.T).float()
		# compute logits
		anchor_dot_contrast = torch.div(
			torch.matmul(proxy_features, contrastive_features.T),
			self.hparams['temperature'])
		# for numerical stability
		logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
		logits = anchor_dot_contrast - logits_max.detach()


		# compute log_prob
		exp_logits = torch.exp(logits)
		log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

		# compute mean of log-likelihood over positive
		mean_log_prob_pos = (label_mask * log_prob).sum(1) / (label_mask.sum(1) + eps)

		# loss
		loss = - mean_log_prob_pos
		loss = loss.mean()
		return loss

 

if __name__ == "__main__":
	la_loss = Local_Alignment_Loss({'resnet18':True, 'k':4, 'projection_size':32})
	fea = torch.rand(64, 512, 7, 7)
	out = la_loss(fea)
	print(out)