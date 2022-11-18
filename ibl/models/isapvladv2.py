import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from sklearn.neighbors import NearestNeighbors

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//16, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//16, kernel_size=1)
        # self.value_conv = nn.Conv2d(
        #     in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        gem_query = self.gem(x) # B x C x 1 x 1
        proj_query = self.query_conv(gem_query).view(m_batchsize,-1,1).permute(0, 2, 1)  # B X 1 X C
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X 1 X HW
        out = torch.bmm(x.view(m_batchsize,C,-1), attention.permute(0, 2, 1)) # B X C X 1
        # proj_value = self.value_conv(x).view(
        #     m_batchsize, -1, width*height)  # B X C X N
        # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = out.view(m_batchsize, C, width, height)
        # out = self.gamma*out + x
        return out.unsqueeze(-1)

class ISAPVLADV2(nn.Module):
    """Iterative self-distillation of attentional pyramid VLAD for Visual Place Recognition"""

    def __init__(self, num_clusters=64, dim=128, pyramid_level=3, overlap=True, shadow_weighting=True, semantic_init=True, num_shadows=4, num_sc=3, param_norm=True,
                 normalize_input=True, vladv2=False, core_loop=True, attentional_pyramid_pooling=True, estimate_uncertainty=False):
        """
        Args:1
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
            pyramid_level: int
                Num of pyramid layers, the k^th layer consists of 4**(k-1) if overlap == False
            Overlap: bool
                If true, using overlapping pyramid vlad pooling on the input feature maps
            Semantic_init: bool
                If true, applying semantic constrained initialization
            Param_norm: bool
                If true, applying parametric normalization
        """
        super(ISAPVLADV2, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input

        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.core_loop = core_loop  # slower than non-looped, but lower memory usage

        # attentional pytramid vlad
        self.pyramid_level = pyramid_level
        self.overlap = overlap
        self.attentional_pyramid_pooling = attentional_pyramid_pooling
        if self.attentional_pyramid_pooling:
            conv_list = []
            for level in range(pyramid_level):
                # h_wid, w_wid, stride_h, stride_w, p2d = self.pyramid_pool_params(
                #     H, W, level)
                conv_list.append(nn.Conv2d(dim, num_clusters,
                                 kernel_size=1, stride=1, bias=False))
            self.conv_app = nn.ModuleList(conv_list)

        # parametric normalization
        self.param_norm = param_norm
        if self.param_norm == True:
            self.cluster_weights = nn.Parameter(torch.ones(num_clusters))

        # ghost weighting
        self.shadow_weighting = shadow_weighting
        self.semantic_init = semantic_init
        if self.shadow_weighting == True:
            # num of shadow centroids should be equal or larger than the num of shadow candidates
            self.num_shadows = max(num_shadows, num_sc)
            self.num_sc = num_sc
            self.conv_shadow = nn.Conv2d(
                dim, self.num_shadows*num_clusters, kernel_size=(1, 1), bias=vladv2)
            # self.dropout = nn.Dropout2d(0.5)
        self.conv = nn.Conv2d(dim, num_clusters,
                              kernel_size=(1, 1), bias=vladv2)

        # self.uc_estimate = uncertainty_estimator # added on 2022 08 05 uncertainty estimator
        # if self.uc_estimate:
        #     self.uc_estimator = []
        #     self.uc_estimator.append(nn.Conv2d(dim, num_clusters,
        #                          kernel_size=1, stride=1, bias=False))

        #2022 08 07 uncertainty estimator
        self.estimate_uncertainty = estimate_uncertainty
        if estimate_uncertainty:
            self.uc_estimator = UC_ESTIMATOR(input_dim = dim, num_clusters = self.num_clusters, num_classes = 2)
        else:
            self.uc_estimator = None
        self.best_uc_estimator = None

        self.clsts=None
        self.traindescs=None
        self.shadowclsts=None

    def init_APPnet(self, clsts=None, xavier_uniform=False):
        if clsts is not None:
            for net_id, net in enumerate(self.conv_app):
                # clst_id = net_id%self.num_clusters
                # clst = clsts[clst_id:clst_id+1] #1,512
                _, _, h, d = net.weight.shape
                net.weight.data.copy_(torch.from_numpy(clsts).unsqueeze(
                    2).unsqueeze(3))  # (64, 512, 1, 1)
        else:
            for net in self.conv_app:
                if xavier_uniform:
                    torch.nn.init.xavier_uniform(net.weight)
                else:
                    net.weight.data.fill_(0.01)

    def _init_params(self):
        assert (self.clsts is not None and self.traindescs is not None and self.shadowclsts is not None), f"=> Cached cluster featuers not found"
        # TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = self.clsts / np.linalg.norm(self.clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, self.traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) /
                          np.mean(dots[0, :] - dots[1, :])).item()
            self.centroids.data.copy_(torch.from_numpy(self.clsts))
            self.conv.weight.data.copy_(torch.from_numpy(
                self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))  # (64, 512, 1, 1)
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1)  # TODO faiss?
            knn.fit(self.traindescs)
            dsSq = np.square(knn.kneighbors(self.clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) /
                          np.mean(dsSq[:, 1] - dsSq[:, 0])).item()
            self.centroids.data.copy_(torch.from_numpy(self.clsts))
            self.conv.weight.data.copy_(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias.data.copy_(
                - self.alpha * self.centroids.norm(dim=1)
            )
        if (self.shadow_weighting == True) and (self.semantic_init == True):
            assert self.shadowclsts is not None, "try to implement semantic constrained initialization but cannot find the shadowclsts"
            clstsAssign = self.clsts/np.linalg.norm(self.clsts, axis=1, keepdims=True)
            clstst = torch.from_numpy(clstsAssign)
            shadowclstsAssign = self.shadowclsts / \
                np.linalg.norm(self.shadowclsts, axis=1, keepdims=True)
            shadowclsts = torch.from_numpy(shadowclstsAssign)
            dis = (shadowclsts.unsqueeze(0) -
                   clstst.unsqueeze(1)).norm(2, 2)  # (64,9)
            candidate_min_dis = dis.sort(
                1, False)[1][:, :self.num_sc]  # min idx (64,3)
            if self.conv.bias is not None:
                shadowconv_bias_sc = - self.alpha * torch.from_numpy(shadowclsts).norm(dim=1)
            for i in range(self.num_clusters):
                # initialize the shadows of ith cluster
                idx_list = list(range(self.num_clusters))
                idx_list.remove(i)
                shadowconv_weight_sc = torch.from_numpy(
                    self.alpha*shadowclstsAssign[candidate_min_dis[i]]).unsqueeze(2).unsqueeze(3)
                if self.num_sc < self.num_shadows:
                    # idx = np.random.choice(idx_list,self.num_shadows-sc_shadow_num,False)
                    idx = np.random.choice(
                        idx_list, self.num_shadows-self.num_sc, False)
                    complshadowconv_weight = self.conv.weight.data[idx].to(shadowconv_weight_sc.device)
                    shadowconv_weight = torch.cat(
                        (shadowconv_weight_sc, complshadowconv_weight), 0)
                    self.conv_shadow.weight.data[i*self.num_shadows:(
                        i+1)*self.num_shadows].copy_(shadowconv_weight)
                    if self.conv.bias is not None:
                        complshadowconv_bias = self.conv.bias.data[idx].to(shadowconv_bias_sc.device)
                        shadowconv_bias = torch.cat(
                            (shadowconv_bias_sc, complshadowconv_bias), 0)
                        self.conv_shadow.bias.data[i*self.num_shadows:(
                            i+1)*self.num_shadows].copy_(shadowconv_bias)
                else:
                    self.conv_shadow.weight.data[i*self.num_shadows:(
                        i+1)*self.num_shadows] = shadowconv_weight_sc
                    if self.conv.bias is not None:
                        self.conv_shadow.bias.data[i*self.num_shadows:(
                            i+1)*self.num_shadows] = shadowconv_bias_sc
        if self.attentional_pyramid_pooling:
            self.init_APPnet(clstsAssign, xavier_uniform=False)
        if self.estimate_uncertainty:
            self.uc_estimator.params_init()

    def pyramid_pool_params(self, H, W, level):
        num_splits = 2**level  # 1, 2, 4, 8
        # N,C,H,W = inpt.shape
        if self.overlap == True:
            h_wid = int(math.ceil(float(2*H) / (num_splits+1)))
            w_wid = int(math.ceil(float(2*W) / (num_splits+1)))
            h_str = int(math.ceil(float(H) / (num_splits+1)))
            w_str = int(math.ceil(float(W) / (num_splits+1)))
            H_new = h_wid + (num_splits - 1)*h_str
            W_new = w_wid + (num_splits - 1)*w_str
            h_pad = int(H_new - H)
            w_pad = int(W_new - W)
            p2d = (int(math.floor(float(w_pad)/2)), int(math.ceil(float(w_pad)/2)),
                   int(math.floor(float(h_pad)/2)), int(math.ceil(float(h_pad)/2)))
        else:
            h_wid = int(math.ceil(float(H) / num_splits))
            w_wid = int(math.ceil(float(W) / num_splits))
            h_str = h_wid
            w_str = w_wid
            H_new = num_splits*h_wid
            W_new = num_splits*w_wid
            h_pad = int(H_new - H)
            w_pad = int(W_new - W)
            p2d = (int(math.floor(float(w_pad)/2)), int(math.ceil(float(w_pad)/2)),
                   int(math.floor(float(h_pad)/2)), int(math.ceil(float(h_pad)/2)))
        return h_wid, w_wid, h_str, w_str, p2d

    def pyramid_aggregation(self, residual, pyramid_saliency=None, C_idx=None):
        N, K, C, H, W = residual.shape
        if self.core_loop == True:
            # operate for one cluster, the residual is in shape of N, 1, C, H, W
            for level in range(self.pyramid_level):
                h_wid, w_wid, stride_h, stride_w, p2d = self.pyramid_pool_params(
                    H, W, level)
                x = F.pad(residual.squeeze(1), p2d, "constant", 0)
                avgpool = nn.AvgPool2d(
                    (h_wid, w_wid), stride=(stride_h, stride_w))
                x = h_wid*w_wid*avgpool(x)  # N, C, split, split
                if level == 0:
                    pyramid_vlad = x.view(N, C, -1)
                else:
                    pyramid_vlad = torch.cat(
                        (pyramid_vlad, x.view(N, C, -1)), 2)

            pyramid_vlad *= pyramid_saliency
            pyramid_vlad = pyramid_vlad.sum(-1)  # N, C
            pyramid_vlad = F.normalize(pyramid_vlad, p=2, dim=1)  # N, C
            return pyramid_vlad.unsqueeze(1)  # N,1,C
        else:
            # operate for K clusters, the residual is in shape of N, K, C, H, W
            print('to be completed')

    def pyramid_saliency(self, x):
        # option2: simple implementation
        N, C, H, W = x.shape

        for level in range(self.pyramid_level):
            h_wid, w_wid, stride_h, stride_w, p2d = self.pyramid_pool_params(
                H, W, level)
            x_pad = F.pad(x, p2d, "constant", 0)
            x_pad = self.conv_app[level](x_pad)  # N,64,h,d
            x_pad = F.relu(x_pad)
            avgpool = nn.AvgPool2d(
                (h_wid, w_wid), stride=(stride_h, stride_w))
            # x_pad = avgpool(x_pad)  # N, C, split, split
            x_pad = h_wid*w_wid*avgpool(x_pad)  # N, C, split, split
            if level == 0:
                pyramid_saliency = x_pad.view(N, self.num_clusters, -1)
            else:
                pyramid_saliency = torch.cat(
                    (pyramid_saliency, x_pad.view(N, self.num_clusters, -1)), 2)

        pyramid_saliency = F.normalize(pyramid_saliency, p=2, dim=2)  # N,K,num
        # pyramid_saliency = F.softmax(pyramid_saliency, dim=2)  # N,K,num
        return pyramid_saliency

    def forward(self, x, vis_kmaps=None, forTNSE=False):

        N, C, H, W = x.shape

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x)  # N, K+n_g, H, W

        shadowvoting = self.conv_shadow(x).view(
            N, self.num_clusters, self.num_shadows, H, W)  # N,K,S,H,W
        # shadowvoting = self.dropout(shadowvoting)
        shadowvoting = torch.cat(
            (soft_assign.unsqueeze(2), shadowvoting), 2)  # N,K,1+S,H,W
        shadowvoting = F.softmax(shadowvoting, dim=2)[:, :, 0, :, :]  # N,K,H,W
        soft_assign = F.softmax(soft_assign, dim=1)
        soft_assign = soft_assign*shadowvoting
        del shadowvoting

        if self.attentional_pyramid_pooling:
            pyramid_saliency = self.pyramid_saliency(x)

        vlad = torch.zeros([N, self.num_clusters, C],
                           dtype=x.dtype, layout=x.layout, device=x.device)
        # calculate residuals to each clusters
        if self.core_loop == True:  # slower than non-looped, but lower memory usage
            for C_idx in range(self.num_clusters):
                residual = x.unsqueeze(0).permute(1, 0, 2, 3, 4) - \
                    self.centroids[C_idx:C_idx+1, :].expand(W, -1, -1).expand(H, -1, -1, -1).permute(
                        2, 3, 0, 1).unsqueeze(0)  # N,1,C,H,W - 1,1,C,H,W = N,1,C,H,W
                # print("residual",residual.shape)('residual', (1, 1, 512, 30, 40))
                # print("centroids", self.centroids.shape)('centroids', (64, 512))
                # N,1,C,H,W * N,1,1,H,W = N,1,C,H,W
                residual *= soft_assign[:, C_idx:C_idx+1, :].unsqueeze(2)

                if self.attentional_pyramid_pooling:
                    vlad[:, C_idx:C_idx+1, :] = self.pyramid_aggregation(
                        residual, pyramid_saliency[:, C_idx:C_idx+1, :], C_idx=C_idx)  # N,K,C
                else:
                    vlad[:, C_idx:C_idx+1,
                         :] = F.normalize(residual.view(N, 1, C, -1).sum(dim=-1), p=2, dim=2)
        else:
            print('to be completed')

        # parametric normalization
        vlad = self.normalization(vlad)
        return vlad

    def forward_patch(self, x):
        return self.forward(x)
        
    def normalization(self, vlad):
        ''' vlad: N, C*K'''
        N = vlad.shape[0]
        # parametric normalization
        if self.param_norm == True:
            cluster_weights = self.cluster_weights.expand(N, -1)
            cluster_weights = F.normalize(cluster_weights, p=2, dim=1)  # N,K
            vlad = vlad*cluster_weights.unsqueeze(2)
        vlad = vlad.view(N, -1)  # flatten  of shape N*(K*C)
        # L2 normalize       #output N*(K*C) tensor
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad

    def infer_uncertianty(self, residual):
        return self.uc_estimator(residual)
