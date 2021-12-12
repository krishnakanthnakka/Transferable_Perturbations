import torch.nn as nn
import torch.nn.functional as F
import torch

from ssd.utils import box_utils


class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(
                loss, labels, self.neg_pos_ratio)

        # --------------------- CHANGED:  PLEASE CHECK THE CODE. UNINTENDED BEHAVIOUR MAY BE THERE -----------------------
        # aadded this since above mask has negative in ratio 3:1, labels>0 or >=0
        mask = labels >= 0

        confidence = confidence[mask, :]
        # print(confidence.shape, labels.shape)

        # for j in range(12):
        #     m1 = labels[j] > 0
        #     print(j, labels[j][m1])

        # exit()

        classification_loss = F.cross_entropy(
            confidence.view(-1, num_classes), labels[mask], reduction='sum')

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(
            predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos
