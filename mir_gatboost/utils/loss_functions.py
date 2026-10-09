import torch
from torch import nn
import torch.nn.functional as F


# Define Focal Loss Calibration
class FocalLoss(nn.Module):
    r"""Unified Focal Loss class for multi-class classification tasks.

    Args:
        gamma (int, optional): Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma. (default: :obj:`1`)
        reduction (str, optional):Specifies the reduction method, possible values are `"mean"`, `"sum"`. (default: :obj:`mean`)
        """
    
    def __init__(self, gamma=1, reduction='mean'):
        
        super(FocalLoss, self).__init__()
        self.gamma = nn.Parameter( torch.Tensor([gamma]) )
        self.reduction = reduction

    def forward(self, logits, labels):
        r"""Forward pass to compute the Focal Loss. 

        Args:
            logits: Predictions (logits) from the model.
                       Shape:
                         - multi-class: (batch_size, num_classes)
            labels: Ground truth labels.
                        Shape:
                         - multi-class: (batch_size)
        """
        
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        # print(f"ce_loss shape -> {ce_loss.shape}") # ------------------------------DEBUG

        # Get the class probabilities for each pixel
        pt = torch.exp(-ce_loss)
        # print(f"pt shape -> {pt.shape}") # ------------------------------DEBUG
        
        
        focal_loss = torch.pow((1 - pt), self.gamma.to(logits.device)) * ce_loss

        # return reducted focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
    
        # return whole focal_loss
        return focal_loss     