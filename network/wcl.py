# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from network.head import *
from network.resnet import *
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np

class WCL(nn.Module):
    def __init__(self, dim_hidden=4096, dim=256):
        super(WCL, self).__init__()
        self.net = resnet50()
        self.head1 = ProjectionHead(dim_in=2048, dim_out=dim, dim_hidden=dim_hidden)
        self.head2 = ProjectionHead(dim_in=2048, dim_out=dim, dim_hidden=dim_hidden)

    @torch.no_grad()
    def build_connected_component(self, dist):
        """
        The build_connected_component function is used to determine the connected components of a graph based on a distance matrix.
        - dist: A square matrix of distances between items, where dist[i, j] represents the distance between item i and item j. It is assumed to be on the GPU.
        """
        b = dist.size(0)
        # This line adjusts the distance matrix dist by subtracting 2 from the diagonal elements. This is done to prevent self-loops from being considered when identifying the closest neighbor.
        dist = dist - torch.eye(b, b, device='cuda') * 2

        # x contains indices [0, 1, ..., b-1] flattened into a single dimension.
        # y contains the indices of the closest neighbor for each item (the index of the smallest non-diagonal element in each row of dist).
        x = torch.arange(b, device='cuda').unsqueeze(1).repeat(1,1).flatten()
        y = torch.topk(dist, 1, dim=1, sorted=False)[1].flatten()
        # rx and ry contain pairs of indices representing edges between nodes in the graph.
        # v is an array of ones representing the weight of the edges.
        # graph is a sparse adjacency matrix (in CSR format) representing the graph, where each node is connected to its closest neighbor.
        rx = torch.cat([x, y]).cpu().numpy()
        ry = torch.cat([y, x]).cpu().numpy()
        v = np.ones(rx.shape[0])
        graph = csr_matrix((v, (rx, ry)), shape=(b,b))
        # connected_components from scipy.sparse.csgraph computes the connected components of the graph. labels assigns each node to a connected component label.
        _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        # labels are converted back to a PyTorch tensor on the GPU.
        # mask is a binary matrix where mask[i, j] is True if items i and j belong to the same connected component and False otherwise.
        labels = torch.tensor(labels, device='cuda')
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T)
        return mask

    def sup_contra(self, logits, mask, diagnal_mask=None):
        """
        Supervised contrastive loss computation
        """
        # exp_logits computes the exponentials of the logits, which are used to convert them into probabilities. 
        if diagnal_mask is not None:
            diagnal_mask = 1 - diagnal_mask
            mask = mask * diagnal_mask
            exp_logits = torch.exp(logits) * diagnal_mask
        else:
            exp_logits = torch.exp(logits)
        # log_prob computes the log probabilities by normalizing exp_logits. This normalization is done by subtracting the log of the sum of exp_logits across the appropriate dimension.
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # mean log probability for positive pairs! This is done by summing the log probabilities of positive pairs (as indicated by the mask) and then dividing by the number of positive pairs.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = (-mean_log_prob_pos).mean()
        return loss

    def forward(self, x1, x2, t=0.1):
        # Distributed Environment Setup
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        # Feature extraction
        b = x1.size(0)
        backbone_feat1 = self.net(x1)
        backbone_feat2 = self.net(x2)
        # normalized after projection head
        feat1 = F.normalize(self.head1(backbone_feat1))
        feat2 = F.normalize(self.head1(backbone_feat2))

        # gather features from other processes
        other1 = concat_other_gather(feat1)
        other2 = concat_other_gather(feat2)

        # Compute Similarity Matrix and Masking
        prob = torch.cat([feat1, feat2]) @ torch.cat([feat1, feat2, other1, other2]).T / t
        # Exclude self-similarities
        diagnal_mask = (1 - torch.eye(prob.size(0), prob.size(1), device='cuda')).bool()
        logits = torch.masked_select(prob, diagnal_mask).reshape(prob.size(0), -1)

        # Creates labels for the concatenated features. This is used to create pairs of positive and negative examples for the contrastive loss.
        first_half_label = torch.arange(b-1, 2*b-1).long().cuda()
        second_half_label = torch.arange(0, b).long().cuda()
        labels = torch.cat([first_half_label, second_half_label])

        # Gather features for graph loss calculation, head2
        feat1 = F.normalize(self.head2(backbone_feat1))
        feat2 = F.normalize(self.head2(backbone_feat2))
        all_feat1 = concat_all_gather(feat1)
        all_feat2 = concat_all_gather(feat2)
        all_bs = all_feat1.size(0)

        # Compute Connectivity Masks - building connected components of the similarity graphs
        mask1_list = []
        mask2_list = []
        if rank == 0:
            mask1 = self.build_connected_component(all_feat1 @ all_feat1.T).float()
            mask2 = self.build_connected_component(all_feat2 @ all_feat2.T).float()
            mask1_list = list(torch.chunk(mask1, world_size))
            mask2_list = list(torch.chunk(mask2, world_size))
            mask1 = mask1_list[0]
            mask2 = mask2_list[0]
        else:
            mask1 = torch.zeros(b, all_bs, device='cuda')
            mask2 = torch.zeros(b, all_bs, device='cuda')
        torch.distributed.scatter(mask1, mask1_list, 0)
        torch.distributed.scatter(mask2, mask2_list, 0)

        diagnal_mask = torch.eye(all_bs, all_bs, device='cuda')
        diagnal_mask = torch.chunk(diagnal_mask, world_size)[rank]
        # Computes the graph loss using the sup_contra function, which is based on the similarity between features and the connectivity masks.
        graph_loss =  self.sup_contra(feat1 @ all_feat1.T / t, mask2, diagnal_mask)
        graph_loss += self.sup_contra(feat2 @ all_feat2.T / t, mask1, diagnal_mask)
        graph_loss /= 2
        return logits, labels, graph_loss



# utils
@torch.no_grad()
def concat_other_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    rank = torch.distributed.get_rank()
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    other = torch.cat(tensors_gather[:rank] + tensors_gather[rank+1:], dim=0)
    return other
    # - rank is the current process’s rank within the distributed group. It is retrieved using torch.distributed.get_rank().
    # - torch.distributed.get_world_size() gives the total number of processes in the distributed group.
    # - tensors_gather is a list of tensors, where each tensor has the same shape and type as the input tensor. The list is initialized to have one tensor for each process in the distributed group.


@torch.no_grad()
def concat_all_gather(tensor, replace=True):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    rank = torch.distributed.get_rank()
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    if replace:
        tensors_gather[rank] = tensor
    other = torch.cat(tensors_gather, dim=0)
    return other


