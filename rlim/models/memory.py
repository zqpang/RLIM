import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy


class Memory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, bg_knn=50):
        super(Memory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        
        self.celoss = CrossEntropyLabelSmooth()
        self.seloss = SoftEntropy()
        
        self.temp = temp
        self.bg_knn = bg_knn

        self.proxy_cam_dict = {}
        self.proxy_label_dict = {}


        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('all_pseudo_label', torch.zeros(num_samples).long())
        self.register_buffer('all_proxy_label', torch.zeros(num_samples).long())
        

    def forward(self, inputs, indexes, cams, num_cluster, epoch):
        # inputs: B*2048, features: L*2048
        
        
        targets = self.all_pseudo_label[indexes].clone()
        proxy_targets = self.all_proxy_label[indexes].clone()
        
        
        
        cluster_sim = torch.matmul(inputs, self.cluster_centers.transpose(1, 0)) / self.temp
        
        loss_cel = self.celoss(cluster_sim, targets, num_cluster)
        
        proxy_sim = torch.matmul(inputs, self.proxy_centers.transpose(1, 0)) / self.temp
        
        offline_loss = 0
        for cc in torch.unique(cams):
                inds = torch.nonzero(cams == cc).squeeze(-1)
                percam_inputs = proxy_sim[inds]
                percam_y = targets[inds]
                
                offline_loss += self.get_proxy_associate_loss(percam_inputs, percam_y)
                
        if epoch+1 >= 0:
            cluster_sim2 = cluster_sim.detach().clone()
            proxy_sim2 = torch.matmul(inputs, self.proxy_centers2.transpose(1, 0)) / self.temp
            
            loss_sel = self.seloss(proxy_sim2, cluster_sim2)
            
            return loss_cel + offline_loss + loss_sel*10
        
        else:
            return loss_cel + offline_loss

    


    def get_proxy_associate_loss(self, inputs, targets):
        temp_inputs = inputs.detach().clone()
        loss = 0
        for i in range(len(inputs)):
            pos_ind = self.proxy_label_dict[int(targets[i])]
            temp_inputs[i, pos_ind] = 10000.0  # mask the positives
            sel_ind = torch.sort(temp_inputs[i])[1][-self.bg_knn-len(pos_ind):]
            sel_input = inputs[i, sel_ind]
            sel_target = torch.zeros((len(sel_input)), dtype=sel_input.dtype).to(torch.device('cuda'))
            sel_target[-len(pos_ind):] = 1.0 / len(pos_ind)
            loss += -1.0 * (F.log_softmax(sel_input.unsqueeze(0), dim=1) * sel_target.unsqueeze(0)).sum()
        loss /= len(inputs)
        return loss
        
        
        
