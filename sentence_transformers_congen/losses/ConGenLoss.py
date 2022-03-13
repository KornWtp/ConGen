import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Dict

class ConGenLoss(nn.Module):
    def __init__(self, instanceQ_encoded, model, teacher_temp=0.1, student_temp=0.09):
        """
        param model:    SentenceTransformerModel
        teacher_temp:   distillation temperature for teacher model 
        student_temp:   distillation temperature for student model 
        """
        super(ConGenLoss, self).__init__()
        self.instanceQ_encoded = instanceQ_encoded
        self.model = model
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

    def forward(self, 
                sents1_features: Iterable[Dict[str, Tensor]],
                sents2_features: Iterable[Dict[str, Tensor]], 
                Z_ref: Tensor):

        # Batch-size
        batch_size = Z_ref.shape[0]

        Z_con = F.normalize(self.model(sents1_features)['sentence_embedding'], p=2, dim=1)
        Z_gen = F.normalize(self.model(sents2_features)['sentence_embedding'], p=2, dim=1)

        # insert the current batch embedding from T
        instanceQ_encoded = self.instanceQ_encoded
        Q = torch.cat((instanceQ_encoded, Z_ref))
    
        # probability scores distribution for T, S: B X (N + 1)
        T_ref = torch.einsum('nc,ck->nk', Z_ref, Q.t().clone().detach())
        S_con = torch.einsum('nc,ck->nk', Z_con, Q.t().clone().detach())
        S_gen = torch.einsum('nc,ck->nk', Z_gen, Q.t().clone().detach())


        # Apply temperatures for soft-labels
        T_ref = F.softmax(T_ref/self.teacher_temp, dim=1)
        S_con = S_con / self.student_temp
        S_gen = S_gen / self.student_temp
        

        # loss computation, use log_softmax for stable computation
        loss_Con = -torch.mul(T_ref, F.log_softmax(S_con, dim=1)).sum() / batch_size
        loss_Gen = -torch.mul(T_ref, F.log_softmax(S_gen, dim=1)).sum() / batch_size
        
        # update the random sample queue
        self.instanceQ_encoded = Q[batch_size:]
  
        return (loss_Con + loss_Gen) / 2