"""
FSS via GMRD
Extended from ADNet code by Hansen et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.encoder import Res101Encoder
import numpy as np
import random
import cv2
from models.moudles import MLP, Decoder

class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.criterion_MSE = nn.MSELoss()
        self.fg_num = 100
        self.bg_num = 600
        self.mlp1 = MLP(256, self.fg_num)
        self.mlp2 = MLP(256, self.bg_num)
        self.decoder1 = Decoder(self.fg_num)
        self.decoder2 = Decoder(self.bg_num)

    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        self.n_ways = len(supp_imgs)  # 1
        self.n_shots = len(supp_imgs[0])  # 1
        self.n_queries = len(qry_imgs)  # 1
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]    # 1
        supp_bs = supp_imgs[0][0].shape[0]      # 1
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)      # (2, 3, 256,256)
        # encoder output
        img_fts, tao = self.encoder(imgs_concat)
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        supp_fts = supp_fts[0]

        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        qry_fts = qry_fts[0]

        ##### Get threshold #######
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        aux_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(supp_bs):
            ###### Extract prototypes ######
            if supp_mask[epi][0].sum() == 0:
                supp_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                               for shot in range(self.n_shots)] for way in range(self.n_ways)]
                fg_prototypes = self.getPrototype(supp_fts_)

                ###### Get query predictions ######
                qry_pred = torch.stack(
                    [self.getPred(qry_fts[epi], fg_prototypes[way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1) # 2 x N x Wa x H' x W'
                preds = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)
                preds = torch.cat((1.0 - preds, preds), dim=1)
                outputs.append(preds)
                if train:
                    align_loss_epi = self.alignLoss([supp_fts[epi]], [qry_fts[epi]], preds, supp_mask[epi])
                    align_loss += align_loss_epi
            else:
                fg_pts = [[self.get_fg_pts(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                                   for shot in range(self.n_shots)] for way in range(self.n_ways)]
                fg_pts = self.get_all_prototypes(fg_pts)     # lsit of list of tensor [(102, 512)]

                bg_pts = [[self.get_bg_pts(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                                   for shot in range(self.n_shots)] for way in range(self.n_ways)]
                bg_pts = self.get_all_prototypes(bg_pts)    # lsit of list of tensor

                ###### Get query predictions ######
                fg_sim = torch.stack(
                    [self.get_fg_sim(qry_fts[epi], fg_pts[way]) for way in range(self.n_ways)], dim=1).squeeze(0)
                bg_sim = torch.stack(
                    [self.get_bg_sim(qry_fts[epi], bg_pts[way]) for way in range(self.n_ways)], dim=1).squeeze(0)

                preds = F.interpolate(fg_sim, size=img_size, mode='bilinear', align_corners=True)
                bg_preds = F.interpolate(bg_sim, size=img_size, mode='bilinear', align_corners=True)

                preds = torch.cat([bg_preds, preds], dim=1)
                preds = torch.softmax(preds, dim=1)

                outputs.append(preds)
                if train:
                    align_loss_epi, aux_loss_epi = self.align_aux_Loss([supp_fts[epi]], [qry_fts[epi]], preds,
                                                         supp_mask[epi], fg_pts, bg_pts)  # fg_pts, bg_pts
                    align_loss += align_loss_epi
                    aux_loss += aux_loss_epi


        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])

        return output, align_loss / supp_bs, aux_loss / supp_bs


    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes
        # (1, 512, 64, 64) (1, 512), (1, 1)
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))   # ([1, 64, 64])

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C] (1, 1, (1, 512))
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  # concat all fg_fts   (n_way, (1, 512))


        return fg_prototypes

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [self.getFeatures(qry_fts[0], pred_mask[way + 1])]
                fg_prototypes = self.getPrototype([qry_fts_])

                # Get predictions
                supp_pred = self.getPred(supp_fts[0][way, [shot]], fg_prototypes[way], self.thresh_pred[way])  # N x Wa x H' x W'
                supp_pred = F.interpolate(supp_pred[None, ...], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                # Combine predictions of different feature maps
                pred_ups = torch.cat((1.0 - supp_pred, supp_pred), dim=1)  # (1, 2, 256, 256)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def align_aux_Loss(self, supp_fts, qry_fts, pred, fore_mask, sup_fg_pts, sup_bg_pts):
        """
            supp_fts: [1, 512, 64, 64]
            qry_fts: (1, 512, 64, 64)
            pred: [1, 2, 256, 256]
            fore_mask: [Way, Shot , 256, 256]

        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        loss_aux = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes         1 x C x H' x W'   1 x H x W
                qry_fts_ = [self.get_fg_pts(qry_fts[0], pred_mask[way + 1])]
                fg_prototypes = self.get_all_prototypes([qry_fts_])
                bg_pts_ = [self.get_bg_pts(qry_fts[0], pred_mask[way + 1])]
                bg_pts_ = self.get_all_prototypes([bg_pts_])

                loss_aux += self.get_aux_loss(sup_fg_pts[way], fg_prototypes[way], sup_bg_pts[way], bg_pts_[way])

                # Get predictions
                supp_pred = self.get_fg_sim(supp_fts[0][way, [shot]], fg_prototypes[way])   # N x Wa x H' x W'
                bg_pred_ = self.get_bg_sim(supp_fts[0][way, [shot]], bg_pts_[way])  # N x Wa x H' x W'
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                bg_pred_ = F.interpolate(bg_pred_, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                # Combine predictions of different feature maps
                preds = torch.cat([bg_pred_, supp_pred], dim=1)
                preds = torch.softmax(preds, dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(preds, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss, loss_aux


    def get_fg_pts(self, features, mask):
        """
        feature: 输入tensor 1 x C x H x W
        mask: 输出tensor 1 x H x W
        prototypes: 输出tensor  k x C
        """
        features_trans = F.interpolate(features, size=mask.shape[-2:], mode='bilinear', align_corners=True)  # [1, 512, 256, 256]
        ie_mask = mask.squeeze(0) - torch.tensor(cv2.erode(mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8), iterations=2)).to(self.device)
        ie_mask = ie_mask.unsqueeze(0)
        ie_prototype = torch.sum(features_trans * ie_mask[None, ...], dim=(-2, -1)) \
                       / (ie_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C
        origin_prototype = torch.sum(features_trans * mask[None, ...], dim=(-2, -1)) \
                         / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        fg_fts = self.get_fg_fts(features_trans, mask)
        fg_prototypes = self.mlp1(fg_fts.view(512, 256*256)).permute(1, 0)
        fg_prototypes = torch.cat([fg_prototypes, origin_prototype, ie_prototype], dim=0)

        return fg_prototypes

    def get_bg_pts(self, features, mask):
        """
        feature: 输入tensor 1 x C x H x W
        mask: 输出tensor H x W
        prototypes: 输出tensor  k x C
        """
        bg_mask = 1 - mask
        features_trans = F.interpolate(features, size=bg_mask.shape[-2:], mode='bilinear', align_corners=True)
        oe_mask = torch.tensor(cv2.dilate(bg_mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8),
                                          iterations=2)).to(self.device) - mask.squeeze(0)
        oe_mask = oe_mask.unsqueeze(0)
        oe_prototype = torch.sum(features_trans * oe_mask[None, ...], dim=(-2, -1)) \
                       / (oe_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C
        origin_prototype = torch.sum(features_trans * bg_mask[None, ...], dim=(-2, -1)) \
                           / (bg_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        bg_fts = self.get_fg_fts(features_trans, bg_mask)
        bg_prototypes = self.mlp2(bg_fts.view(512, 256 * 256)).permute(1, 0)
        bg_prototypes = torch.cat([bg_prototypes, origin_prototype, oe_prototype], dim=0)

        return bg_prototypes

    def get_random_pts(self, features_trans, mask, n_protptype):

        features_trans = features_trans.squeeze(0)
        features_trans = features_trans.permute(1, 2, 0)
        features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
                                             features_trans.shape[-1])
        mask = mask.squeeze(0).view(-1)
        indx = mask == 1
        features_trans = features_trans[indx]  # (n_fg x 512)
        if len(features_trans) >= n_protptype:
            k = random.sample(range(len(features_trans)), n_protptype)
            prototypes = features_trans[k]
        else:
            if len(features_trans) == 0:
                prototypes = torch.zeros(n_protptype, 512).to(self.device)
            else:
                r = (n_protptype) // len(features_trans)
                k = random.sample(range(len(features_trans)), (n_protptype - len(features_trans)) % len(features_trans))
                prototypes = torch.cat([features_trans for _ in range(r)], dim=0)
                prototypes = torch.cat([features_trans[k], prototypes], dim=0)

        return prototypes  # (n_prototype, 512)

    def get_fg_fts(self, fts, mask):
        _, c, h, w = fts.shape
        # select masked fg features
        fg_fts = fts * mask[None, ...]
        bg_fts = torch.ones_like(fts) * mask[None, ...]
        mask_ = mask.view(-1)
        n_pts = len(mask_) - len(mask_[mask_ == 1])
        select_pts = self.get_random_pts(fts, mask, n_pts)    # (n_pts x 512)
        index = bg_fts == 0
        fg_fts[index] = select_pts.permute(1, 0).reshape(512*n_pts)

        return fg_fts


    def get_all_prototypes(self, fg_fts):
        """
            fg_fts: lists of list of tensor
                        expect shape: Wa x Sh x [all x C]
            fg_prototypes: [(all, 512) * way]    list of tensor
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        prototypes = [sum([shot for shot in way]) / n_shots for way in fg_fts]
        return prototypes


    def get_fg_sim(self, fts, prototypes):
        """
        Calculate the distance between features and prototypes
        # (1, 512, 64, 64) (102, 512)
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        fts_ = fts.permute(0, 2, 3, 1)
        fts_ = F.normalize(fts_, dim=-1)
        pts_ = F.normalize(prototypes, dim=-1)
        fg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(0, 3, 1, 2)
        fg_sim = self.decoder1(fg_sim)

        return fg_sim   # [1, 1, 64, 64]

    def get_bg_sim(self, fts, prototypes):
        """
        Calculate the distance between features and prototypes
        # (1, 512, 64, 64) (102, 512)
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        fts_ = fts.permute(0, 2, 3, 1)
        fts_ = F.normalize(fts_, dim=-1)
        pts_ = F.normalize(prototypes, dim=-1)
        bg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(0, 3, 1, 2)
        bg_sim = self.decoder2(bg_sim)

        return bg_sim  # [1, 1, 64, 64]

    def get_aux_loss(self, sup_fg_pts, qry_fg_pts, sup_bg_pts, qry_bg_pts):
        d1 = F.normalize(sup_fg_pts, dim=-1)
        d2 = F.normalize(qry_fg_pts, dim=-1)
        b1 = F.normalize(sup_bg_pts, dim=-1)
        b2 = F.normalize(qry_bg_pts, dim=-1)

        fg_intra0 = torch.matmul(d1[:-2], d2[:-2].transpose(0, 1))
        fg_intra1 = torch.matmul(d1[[-2]], d2[-2])
        fg_intra2 = torch.matmul(d1[[-1]], d2[-1])
        fg_intra0, _ = torch.max(fg_intra0, dim=1)
        fg_intra0 = torch.cat([fg_intra0, fg_intra1, fg_intra2])
        fg_intra0 = torch.mean(fg_intra0)

        bg_intra0 = torch.matmul(b1[:-2], b2[:-2].transpose(0, 1))
        bg_intra1 = torch.matmul(b1[[-2]], b2[-2])
        bg_intra2 = torch.matmul(b1[[-1]], b2[-1])
        bg_intra0, _ = torch.max(bg_intra0, dim=1)
        bg_intra0 = torch.cat([bg_intra0, bg_intra1, bg_intra2])
        bg_intra0 = torch.mean(bg_intra0)
        intra_loss = 2 - fg_intra0 - bg_intra0

        sup_inter = torch.matmul(d1, b1.transpose(0, 1))
        qry_inter = torch.matmul(d2, b2.transpose(0, 1))
        inter_loss = max((0, torch.mean(sup_inter))) + max((0, torch.mean(qry_inter)))

        return intra_loss + inter_loss







