
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pdb

"""================================================================================================="""
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = False




        
class Criterion(torch.nn.Module):

    def __init__(self, opt):
        super(Criterion, self).__init__()

        assert(opt.bs%opt.samples_per_class==0)
        self.b = opt.b
        self.alpha = opt.alpha
        self.anneal = opt.anneal
        self.variant = opt.variant
        self.batch_size = opt.bs
        self.num_id = int(opt.bs/opt.samples_per_class)
        self.samples_per_class = opt.samples_per_class
        self.feat_dims = opt.embed_dim
        self.name           = 'PNP'
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM
        mask = 1.0 - torch.eye(self.batch_size)
        for i in range(self.num_id):
            mask[i*(self.samples_per_class):(i+1)*(self.samples_per_class),i*(self.samples_per_class):(i+1)*(self.samples_per_class)] = 0
        
        self.mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1).cuda()
    def forward(self, batch, labels, **kwargs):
        #if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        sim_all = compute_aff(batch)
        sim_all = sim_all 
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the sigmoid and ignores the relevance score of the query to itself
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * self.mask.cuda()
        # compute the rankings,all batch
        sim_all_rk = torch.sum(sim_sg, dim=-1) 
        if self.variant == 'PNP-D_s':
            sim_all_rk = torch.log(1+sim_all_rk)
        elif self.variant == 'PNP-D_q':
            sim_all_rk = 1/(1+sim_all_rk)**(self.alpha)


        elif self.variant == 'PNP-I_u':
            sim_all_rk = (1+sim_all_rk)*torch.log(1+sim_all_rk)
            
        elif self.variant == 'PNP-I_b':
            b = self.b
            sim_all_rk = 1/b**2 * (b*sim_all_rk-torch.log(1+b*sim_all_rk))
        elif self.variant == 'PNP-O':
            pass
        else:
                raise Exception('variantation <{}> not available!'.format(self.variant))
        
        
        # sum the values of the Smooth-AP for all instances in the mini-batch
        loss = torch.zeros(1).cuda()
        group = int(self.batch_size / self.num_id)
        
        

        for ind in range(self.num_id):
            
            neg_divide = torch.sum(sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)])
            
            loss = loss + ((neg_divide / group) / self.batch_size)
        if  self.variant == 'PNP-D_q':
            return 1 - loss
        else:
            return loss


def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


def compute_aff(x):
    """computes the affinity matrix between an input vector and itself"""
    return torch.mm(x, x.t())
