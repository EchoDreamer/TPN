import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class ATLoss_cal(nn.Module):
    def __init__(self, gamma_pos=1):
        super().__init__()
        self.gamma_pos=gamma_pos

    def forward(self, logits, labels):

        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0
        n_mask = 1 - labels
        _, num_class = labels.size()
        th_mask = torch.cat( num_class * [logits[:,:1]], dim=1)
        logit_th = torch.cat([logits.unsqueeze(1), 1.0 * th_mask.unsqueeze(1)], dim=1) 
        log_probs = F.log_softmax(logit_th, dim=1)
        probs = torch.exp(F.log_softmax(logit_th, dim=1))
        prob_0 = probs[:, 1 ,:]
        prob_0_gamma = torch.pow(prob_0, self.gamma_pos)
        log_prob_1 = log_probs[:, 0 ,:]
        logit2 = logits - (1 - n_mask) * 100
        rank2 = F.log_softmax(logit2, dim=-1)
        loss1 = - (log_prob_1 * (1 + prob_0_gamma ) * labels) 
        loss2 = -(rank2 * th_label).sum(1) 
        loss =  1.0 * loss1.sum(1).mean() + 4.0 * loss2.mean()
        return loss

    def get_label(self, logits, num_labels=-1):     
        th_logit = logits[:, 0].unsqueeze(1)

        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        mask.type(torch.bool)
        output[mask] = 1.0
        
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

