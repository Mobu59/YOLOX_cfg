#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import numpy as np 


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        #pred target:[cx, cy, w, h]
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2

        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == 'diou':
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1], 2) + 1e-7
            center_dis = torch.pow(pred[:, 0] - target[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1], 2)
            diou = 1 - (center_dis / convex_dis)
            loss = 1 - diou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == 'ciou':
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1], 2) + 1e-7
            center_dis = torch.pow(pred[:, 0] - target[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1], 2)
            v = (4 / np.pi ** 2) * torch.pow(torch.atan(target[:, 2] / torch.clamp(target[:, 3], min=1e-7)) - torch.atan(pred[:, 2] / torch.clamp(target[:, 3], min=1e-7)), 2)
            with torch.no_grad():        
                alpha = v / ((1 + 1e-7) - iou + v)
            ciou = iou - (center_dis / convex_dis + alpha * v)    
            loss = 1 - ciou.clamp(min=-1.0, max=1.0)
        
        elif self.loss_type == "siou":
            """
            SIOU Loss:https://readpaper.com/pdf-annotate/note?noteId=694047614248329216&pdfId=4627766541656539137
            SIOU = IOU - (distance_cost + shape_cost) / 2
            Args:
                c_tl:最小外接矩形左上角的点坐标
                c_br:最小外接矩形右下角的点坐标
                cw:预测框和真实框中心点的横坐标距离
                ch:预测框和真实框中心点的纵坐标距离
                alpha:待优化的角
            """
            #get min_tl, max_br
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            #get min enclosing w and h
            s_cw = (c_br - c_tl)[:, 0]
            s_ch = (c_br - c_tl)[:, 1]
            #get center distance
            cw = target[:, 0] - pred[:, 0]
            ch = target[:, 1] - pred[:, 1]
            sigma = torch.pow(cw ** 2 + ch ** 2, 0.5) 
            #sinα
            sin_alpha = torch.abs(ch) / sigma 
            #sinβ
            sin_beta = torch.abs(cw) / sigma 
            #threshold π/4, if α<=π/4，optmize α, else β
            thres = torch.pow(torch.tensor(2.), 0.5) / 2
            sin_alpha = torch.where(sin_alpha < thres, sin_alpha, sin_beta)
            angle_cost = 1 - 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha) - np.pi/4), 2)
            #angle_cost = 2 * sin_alpha * cos_alpha = 2 * sin_alpha * sin_beta
            
            #get diatance cost
            gamma = angle_cost - 2
            rho_x = (cw / s_cw) ** 2
            rho_y = (ch / s_ch) ** 2
            delta_x = 1 - torch.exp(gamma * rho_x) 
            delta_y = 1 - torch.exp(gamma * rho_y)
            distance_cost = delta_x + delta_y
            
            #get shape cost
            w_gt = target[:, 2]
            h_gt = target[:, 3]
            w_pred = pred[:, 2]
            h_pred = pred[:, 3]
            W_w = torch.abs(w_pred - w_gt) / torch.max(w_pred, w_gt)
            W_h = torch.abs(h_pred - h_gt) / torch.max(h_pred, h_gt)
            #hyper parameter θ
            theta = 4
            shape_cost = torch.pow((1 - torch.exp(-1 * W_w)), theta) + torch.pow((1 - torch.exp(-1 * W_h)), theta)
            siou = iou - (distance_cost + shape_cost) * 0.5
            loss = 1 - siou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class alpha_IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="ciou", alpha=3):
        super(alpha_IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type
        self.alpha = alpha

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        #tl:top left, br:bottom right
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )
        #torch.prod means tensor element-wise product    
        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** self.alpha
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou * self.alpha - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == 'ciou':
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1], 2) + 1e-7
            center_dis = torch.pow(pred[:, 0] - target[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1], 2)
            v = (4 / np.pi ** 2) * torch.pow(torch.atan(target[:, 2] / torch.clamp(target[:, 3], min=1e-7)) - torch.atan(pred[:, 2] / torch.clamp(target[:, 3], min=1e-7)), 2)
            with torch.no_grad():        
                beat = v / (v - iou + 1)
            ciou = iou ** self.alpha - (center_dis ** self.alpha / convex_dis
                    ** self.alpha + (beat * v) ** self.alpha)    
            loss = 1 - ciou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

if __name__ == "__main__":
    a = torch.tensor([[0,0,100,100],[0,0,100,100]])
    b = torch.tensor([[200,0,300,100],[200,0,300,100]])
    loss = IOUloss(loss_type="siou")
    print(loss(a, b))
