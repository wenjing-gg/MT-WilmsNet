import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------- 1) Classification main task loss: label smoothing cross entropy ------------------- #
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, target):
        num_classes = pred.size(1)
        target = target.long()
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        target_smoothed = (1 - self.epsilon) * one_hot + self.epsilon / num_classes
        
        log_pred = F.log_softmax(pred, dim=1)
        loss = -torch.sum(target_smoothed * log_pred, dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ------------------- 2) Self-distillation loss function ------------------- #
def self_distillation_loss(teacher_logits, student_logits_list, T=4.0):
    """
    teacher_logits: main classifier output (B, num_classes)
    student_logits_list: Indicates the output list of a shallow classifier
    T: Distillation temperature
    """
    teacher_log_prob = F.log_softmax(teacher_logits / T, dim=1)
    teacher_prob = teacher_log_prob.exp()

    distill_loss_sum = 0.0
    for s_logits in student_logits_list:
        student_log_prob = F.log_softmax(s_logits / T, dim=1)
        distill_kld = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean')
        distill_loss_sum += distill_kld

    distill_loss_sum = distill_loss_sum * (T ** 2)
    return distill_loss_sum

class SoftDiceLoss3D(nn.Module):
    """
    dichotomy 3D Soft Dice Loss
    seg_logits: [B, 1, D, H, W] (logits)
    seg_labels: [B, 1, D, H, W] (0 or 1, float)
    """
    def __init__(self, eps=1e-6):
        super(SoftDiceLoss3D, self).__init__()
        self.eps = eps

    def forward(self, seg_logits, seg_labels):
        # 1) logits -> [0,1] probability
        probs = torch.sigmoid(seg_logits)  # [B, 1, D, H, W]

        # 2) Flatten D*H*W together to become [B, D*H*W]
        B = probs.shape[0]
        probs_flat = probs.view(B, -1)  
        labels_flat = seg_labels.view(B, -1)

        # 3) Calculate intersection and denominator (intersection/union)
        intersection = (probs_flat * labels_flat).sum(dim=1)
        denominator  = probs_flat.sum(dim=1) + labels_flat.sum(dim=1) + self.eps

        # 4) Dice coefficients were calculated separately for each batch sample and then averaged
        dice = 2.0 * intersection / denominator  # [B]
        dice_loss = 1.0 - dice  # [B]

        return dice_loss.mean()  # Scalar quantity

class BCEDiceLoss3D(nn.Module):
    """
    Combined loss of binary cross entropy (BCE) and Soft Dice in 3D binary classification segmentation scenario.

    Applicable to:
    seg_logits: [B, 1, D, H, W] (pure logits, without Sigmoid)
    seg_labels: [B, 1, D, H, W], 0 or 1 (float or convertible to float)

    Algorithm:
    1) Do BCE with Logits Loss on seg_logits first
    2) Then Sigmoid seg_logits and calculate binary Dice
    3) Final weighted merge (alpha*BCE + beta*Dice)
    """
    def __init__(self, alpha=1.0, beta=1.0, eps=1e-6):
        """
        :param alpha: Loss weight of BCE
        :param beta: Dice loss weight
        :param eps: A tiny constant that prevents the denominator from being 0
        """
        super(BCEDiceLoss3D, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        # PyTorch has its own binary cross entropy (including Sigmoid), and can also write its own F.Bian cross_entropy_with_logits
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, seg_logits, seg_labels):
        """
        :param seg_logits: [B, 1, D, H, W] (float, logits)
        :param seg_labels: [B, 1, D, H, W], 0/1 (float or long)
        :return: Combined total loss (scalar)
        """
        # If seg_labels is an integer, it must be changed to float
        if seg_labels.dtype is not torch.float32:
            seg_labels = seg_labels.float()

        # 1) Binary cross entropy loss (BCE), internal automatic log(sigmoid(x))
        bce = self.bce_loss(seg_logits, seg_labels)

        # 2) Dice loss
        #   Start by logits -> [0,1] probability
        probs = torch.sigmoid(seg_logits)  # [B, 1, D, H, W]
        
        #   Flatten: Merge D*H*W, change [B, -1]
        B = probs.shape[0]
        probs_flat = probs.view(B, -1)       # [B, D*H*W]
        labels_flat = seg_labels.view(B, -1) # [B, D*H*W]

        intersection = (probs_flat * labels_flat).sum(dim=1)  # [B]
        denominator  = probs_flat.sum(dim=1) + labels_flat.sum(dim=1) + self.eps
        dice_per_batch = 2.0 * intersection / denominator      # [B]
        dice_mean = dice_per_batch.mean()                      # Scalar quantity
        dice_loss = 1.0 - dice_mean

        # 3) Ultimate loss: alpha * BCE + beta * Dice
        total_loss = self.alpha * bce + self.beta * dice_loss
        return total_loss

class CEDiceLoss3D(nn.Module):
    """
    Combined loss of cross entropy (CE) and Soft Dice in 3D multi-classification segmentation scenario.

    Applicable to:
    seg_logits: [B, C, D, H, W] (unnormalized logits, where C is the number of classes)
    seg_labels: [B, D, H, W], an integer ranging from 0 to C-1 (long)

    Algorithm:
    1) CrossEntropyLoss for seg_logits first
    2) Do Softmax for seg_logits and calculate multi-class Dice
    3) Final weighted merge (alpha*CE + beta*Dice)
    """
    def __init__(self, alpha=1.0, beta=1.0, eps=1e-6, num_classes=2):
        """
        :param alpha: CE loss weight
        :param beta: Dice loss weight
        :param eps: A tiny constant that prevents the denominator from being zero
        :param num_classes: Split class number
        """
        super(CEDiceLoss3D, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, seg_logits, seg_labels):
        """
        :param seg_logits: [B, C, D, H, W] (float, logits)
        :param seg_labels: [B, D, H, W], 0~C-1 (long)
        :return: Combined total loss (scalar)
        """
        # Make sure the label is of type long
        assert seg_labels.dtype == torch.long, "seg_labels must be long type"

        # 1) Cross entropy loss (CE)
        ce = self.ce_loss(seg_logits, seg_labels)

        # 2) Multiple classification Dice loss
        # Convert logits to probabilities [B, C, D, H, W]
        probs = torch.softmax(seg_logits, dim=1)
        
        # Convert tags to one-hot encoding [B, C, D, H, W]
        seg_labels_onehot = F.one_hot(seg_labels, self.num_classes).permute(0, 4, 1, 2, 3)
        
        # Flattened dimension [B, C, D*H*W]
        B, C = probs.shape[0], probs.shape[1]
        probs_flat = probs.view(B, C, -1)             # [B, C, D*H*W]
        labels_flat = seg_labels_onehot.view(B, C, -1)  # [B, C, D*H*W]

        # Calculate the Dice for each category
        intersection = (probs_flat * labels_flat).sum(dim=2)  # [B, C]
        denominator = probs_flat.sum(dim=2) + labels_flat.sum(dim=2) + self.eps  # [B, C]
        dice_per_class = 2.0 * intersection / denominator    # [B, C]
        dice_mean = dice_per_class.mean()                     # Scalar (average of all classes)
        dice_loss = 1.0 - dice_mean

        # 3) Ultimate loss: alpha * CE + beta * Dice
        total_loss = self.alpha * ce + self.beta * dice_loss
        return total_loss