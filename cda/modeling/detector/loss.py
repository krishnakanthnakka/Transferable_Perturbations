
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


def attention_loss(clean_activations, adv_activations, criterionAT):

    att_loss = 0.0
    for key in clean_activations.keys():
        if key == 'act1':
            att_loss += criterionAT(clean_activations[key],
                                    adv_activations[key])

    return att_loss


def feat_loss(adv_activations, clean_activations, criterion, SOFTMAX_2D):

    # loss = criterion(clean_activations['fc'], adv_activations['fc'])

    # loss = criterion(F.normalize(
    #     clean_activations['fc'], p=2, dim=1), F.normalize(adv_activations['fc'], p=2, dim=1))

    # loss = criterion(torch.mean(clean_activations['fc'],
    #                             (2, 3), keepdims=True), adv_activations['fc'])

    # margin = 20
    # feat_fc = clean_activations['fc'].view(clean_activations['fc'].shape[0], -1)
    # feat_adv = adv_activations['fc'].view(adv_activations['fc'].shape[0], -1)
    # loss = torch.mean((feat_fc - feat_adv)**2 * (torch.abs(feat_fc - feat_adv) < margin))

    # feat_fc = clean_activations['fc'].view(clean_activations['fc'].shape[0], -1)
    # feat_adv = adv_activations['fc'].view(adv_activations['fc'].shape[0], -1)
    # num = torch.sqrt(torch.sum((feat_fc - feat_adv)**2, 1))
    # denom = torch.sqrt(torch.sum((feat_fc)**2, 1))
    # loss = torch.mean(num / denom)

    # print(loss1, num.shape, loss, denom, num / denom)
    # exit()

    # print('fc', loss)

    loss = torch.tensor(0.0).cuda()
    alpha = 1

    for key in clean_activations.keys():
        if 'feat' in key:

            # print(key)
            loss_layer = criterion(alpha * clean_activations[key], alpha * adv_activations[key])

            # loss_layer = criterion(clean_activations[key].mean(
            #     (2, 3)), adv_activations[key].mean((2, 3)))

            # B, C, H, W = clean_activations[key].shape
            # loss_layer = gram_loss(clean_activations[key], adv_activations[key], C, H * W * B)

            # print(key, loss_layer)
            loss += 1 * loss_layer

    # print(key, clean_activations[key].shape)
    # exit()
    # loss = criterion(F.normalize(
    #     clean_activations['fc'], p=2, dim=-1), F.normalize(adv_activations['fc'], p=2, dim=-1))

    return loss


def gram(tensor):
    return torch.mm(tensor, tensor.t())


def gram_loss(noise_img_gram, style_img_gram, N, M):
    return torch.sum(torch.pow(noise_img_gram - style_img_gram, 2)).div((np.power(N * M * 2, 2, dtype=np.float64)))


def feat_loss_mutliscale_fn(clean_activations, adv_activations, criterion, SOFTMAX_2D):

    loss = torch.tensor(0.0).cuda()
    # loss = criterion(clean_activations['fc'], adv_activations['fc'])

    softmax = nn.Softmax2d()
    softmax1D = nn.Softmax(dim=-1)
    alpha = 1.0
    # SOFTMAX_2D = True

    for key in clean_activations.keys():
        if 'feat' in key:

            B = clean_activations[key].shape[0]

            if SOFTMAX_2D:
                # loss_layer = criterion(
                #     softmax(clean_activations[key]), softmax(adv_activations[key]))

                loss_layer = criterion(
                    softmax1D(clean_activations[key].view(B, -1)), softmax1D(adv_activations[key].view(B, -1)))

            else:
                loss_layer = criterion(alpha * clean_activations[key], alpha * adv_activations[key])

                # margin = 1
                # feat_fc = clean_activations[key].view(B, -1)
                # feat_adv = adv_activations[key].view(B, -1)
                # loss_layer = torch.mean(((feat_fc - feat_adv)**2) *
                #                         (torch.abs(feat_fc - feat_adv) < margin))

                # loss_layer = torch.mean((torch.abs(feat_fc - feat_adv)**1) *
                #                         (torch.abs(feat_fc - feat_adv) < margin))

            # loss_layer = criterion(clean_activations[key].mean(
            #     (2, 3)), adv_activations[key].mean((2, 3)))

            loss += loss_layer

    return loss


def loss_gram_mutliscale(clean_activations, adv_activations, criterion):

    loss = torch.tensor(0.0).cuda()
    # loss = criterion(clean_activations['fc'], adv_activations['fc'])

    # loss = criterion(F.normalize(
    #     clean_activations['fc'], p=2, dim=1), F.normalize(adv_activations['fc'], p=2, dim=1))

    # loss = criterion(torch.mean(clean_activations['fc'],
    #                             (2, 3), keepdims=True), adv_activations['fc'])

    # margin = 20
    # feat_fc = clean_activations['fc'].view(clean_activations['fc'].shape[0], -1)
    # feat_adv = adv_activations['fc'].view(adv_activations['fc'].shape[0], -1)
    # loss = torch.mean((feat_fc - feat_adv)**2 * (torch.abs(feat_fc - feat_adv) < margin))

    # feat_fc = clean_activations['fc'].view(clean_activations['fc'].shape[0], -1)
    # feat_adv = adv_activations['fc'].view(adv_activations['fc'].shape[0], -1)
    # num = torch.sqrt(torch.sum((feat_fc - feat_adv)**2, 1))
    # denom = torch.sqrt(torch.sum((feat_fc)**2, 1))
    # loss = torch.mean(num / denom)

    # print(loss1, num.shape, loss, denom, num / denom)
    # exit()

    # print('fc', loss)

    # for key in clean_activations.keys():
    #     if 'feat' in key:
    #         loss_layer = criterion(clean_activations[key], adv_activations[key])
    #         loss += loss_layer

    for key in clean_activations.keys():
        if 'fc' in key:
            B, C, H, W = clean_activations[key].shape
            print(key, clean_activations[key].shape)
            gram_clean = gram(clean_activations[key].squeeze().view(C, H * W))
            # print(gram_clean.shape)
            gram_adv = gram(adv_activations[key].squeeze().view(C, H * W))
            loss_layer = gram_loss(gram_clean, gram_adv, C, H * W) / 5.
            loss += loss_layer

    return loss


def gram_bmm(tensor):
    return torch.bmm(tensor, tensor.permute(0, 2, 1))


def gram_bmm_power(tensor, power):

    tensor = tensor**power
    tensor = torch.bmm(tensor, tensor.permute(0, 2, 1))
    # tensor = (tensor.sign() * torch.abs(tensor)**(1 / power))
    return tensor


def gram_loss_multiscale_fn(clean_activations, adv_activations, criterion, SOFTMAX_2D):

    softmax = nn.Softmax2d()

    loss = torch.tensor(0.0).cuda()
    for key in clean_activations.keys():
        if 'feat' in key:
            B, C, H, W = clean_activations[key].shape

            if SOFTMAX_2D:
                clean_activations[key] = softmax(clean_activations[key])
                adv_activations[key] = softmax(adv_activations[key])

            feat_clean = clean_activations[key].view(B, C, H * W)
            feat_adv = adv_activations[key].view(B, C, H * W)

            gram_clean = gram_bmm_power(feat_clean, 1)
            gram_adv = gram_bmm_power(feat_adv, 1)
            loss_layer = gram_loss(gram_clean, gram_adv, C, H * W)
            loss += loss_layer

    loss = loss / len(clean_activations.keys())

    return loss
