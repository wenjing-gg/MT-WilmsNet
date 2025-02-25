import os
import sys
import numpy as np
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,roc_curve, f1_score, precision_score, recall_score
import seaborn as sns
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.amp as amp  # Introduce modules required for mixed precision training
from utils.loss import LabelSmoothingCrossEntropy, self_distillation_loss, BCEDiceLoss3D, CEDiceLoss3D

import scipy.ndimage as ndi

def assd(pred, gt):
    """
    Calculate the Average Symmetric Surface Distance (ASSD).
    Works with 2D or 3D binary masks and increases validity checks for input and output results.

    Args:
    -------
    pred : np.ndarray
        The predicted binary mask, which takes the value {0,1}, can be 2D or 3D.
    gt   : np.ndarray
        The real binary mask is {0,1} and can be 2D or 3D.

    Returns:
    -------
    asd : float
        The average symmetric surface distance (ASSD) returns 0.0 if an exception occurs
    """
    # =============== 1) Input validity check ===============
    # 1.1) Whether the shape is consistent
    if pred.shape != gt.shape:
        raise ValueError(f"Predict the mask shape {pred.shape} and real mask shape {gt.shape} is not same")
    
    # 1.2) Whether the dimension is 2D or 3D
    if pred.ndim not in (2, 3):
        raise ValueError(f"Only 2D or 3D data is supported, but the current input dimension is {pred.ndim}D。")

    # 1.3) Convert the value to bool and check whether the value is 0 or 1
    pred_unique = np.unique(pred)
    gt_unique = np.unique(gt)
    if not set(pred_unique).issubset({0, 1}) or not set(gt_unique).issubset({0, 1}):
        raise ValueError("The input mask should contain only binary values {0, 1}.")
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    
    # =============== 2) Special case treatment ===============
    # If both are empty (all zeros), the distance is 0
    if not pred.any() and not gt.any():
        return 0.0

    # =============== 3) Calculated distance ===============
    try:
        # 3.1) Extract the respective boundary points
        pred_surface = _extract_surface(pred)
        gt_surface   = _extract_surface(gt)

        # 3.2) Calculate the distance transform
        dist_pred = ndi.distance_transform_edt(~pred)
        dist_gt   = ndi.distance_transform_edt(~gt)

        # 3.3) Calculate the distance from the pred boundary to gt, and the distance from the gt boundary to pred
        dists_pred_to_gt = dist_gt[pred_surface]
        dists_gt_to_pred = dist_pred[gt_surface]

        # 3.4) Average them separately, and then average them both
        asd_pred_to_gt = dists_pred_to_gt.mean()  # Average distance (predicted boundary to GT)
        asd_gt_to_pred = dists_gt_to_pred.mean()  # Average distance (GT boundary to forecast)
        asd = (asd_pred_to_gt + asd_gt_to_pred) / 2.0

    except Exception as e:
        print(f"[Warning] ASSD Calculation exception:{e}")
        return 0.0  # Or return np.nan, depending on your requirements
    
    # =============== 4) Output check and return ===============
    # If the result is NaN or negative (theoretically no negative values, but just in case)
    if np.isnan(asd) or asd < 0:
        print("[Warning] ASSD The result of the calculation is NaN or negative, which is automatically set to 0.0")
        asd = 0.0
    
    return float(asd)


def hd95(pred, gt):
    """
    Calculate the 95% Hausdorff distance (HD95).
    Works with 2D or 3D binary masks and increases validity checks for input and output results.

    Args:
    -------
        pred : np.ndarray
            The predicted binary mask, which takes the value {0,1}, can be 2D or 3D.
        gt   : np.ndarray
            The real binary mask is {0,1} and can be 2D or 3D.

    Return:
    -------
        hd_95 : float
        95% Hausdorff distance, returning 0.0 if an exception occurs
    """
    # =============== 1) Whether the shape is consistent ===============
    if pred.shape != gt.shape:
        raise ValueError(f"Predict the mask shape {pred.shape} and real mask shape {gt.shape} is not same")
    if pred.ndim not in (2, 3):
        raise ValueError(f"Only 2D or 3D data is supported, but the current input dimension is {pred.ndim}D。")

    pred_unique = np.unique(pred)
    gt_unique = np.unique(gt)
    if not set(pred_unique).issubset({0, 1}) or not set(gt_unique).issubset({0, 1}):
        raise ValueError("The input mask should contain only binary values {0, 1}.")
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    if not pred.any() and not gt.any():
        return 0.0

    # =============== 2) Computing HD95 ===============
    try:
        pred_surface = _extract_surface(pred)
        gt_surface   = _extract_surface(gt)

        dist_pred = ndi.distance_transform_edt(~pred)
        dist_gt   = ndi.distance_transform_edt(~gt)

        dists_pred_to_gt = dist_gt[pred_surface]
        dists_gt_to_pred = dist_pred[gt_surface]

        # Combine the distance and take 95% quantile
        all_dist = np.concatenate([dists_pred_to_gt, dists_gt_to_pred], axis=None)
        hd_95 = np.percentile(all_dist, 95)
    except Exception as e:
        print(f"[Warning] HD95 Calculation exception:{e}")
        return 0.0  # Or return np.nan, depending on your requirements

    # =============== 3) Output check and return ===============
    if np.isnan(hd_95) or hd_95 < 0:
        print("[Warning] HD95 The result of the calculation is NaN or negative, which is automatically set to 0.0")
        hd_95 = 0.0

    return float(hd_95)


def _extract_surface(mask):
    """
    The boundary point (surface) of the binary mask is extracted.
    mask is a 2D/3D array of type bool.

    Idea: The mask does XOR with the result after corrosion, that is, the boundary position is obtained.
    The final value returned is a tuple (idx1, idx2,...). Can be used to index to mask.
    """
    eroded_mask = ndi.binary_erosion(mask)
    surface = np.logical_xor(mask, eroded_mask)
    return np.where(surface)



# ------------------- 3) Training function for each epoch (classification main, segmentation auxiliary) ------------------- #
def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    scaler=None,
    accumulation_steps=4,
    distill_T=4.0
):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    criterion_main = LabelSmoothingCrossEntropy(epsilon=0.1, reduction='mean')
    seg_criterion = CEDiceLoss3D()
    criterion_aux = LabelSmoothingCrossEntropy(epsilon=0.1, reduction='mean')

    data_loader = tqdm(data_loader, file=sys.stdout)
    data_loader.set_description(f"Training Epoch {epoch}")

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(data_loader):
        imgs, class_labels, seg_labels = batch
        imgs = imgs.to(device)
        class_labels = class_labels.to(device)
        seg_labels = seg_labels.to(device)  # [B, 1 ,D, H, W]
        seg_labels = seg_labels.squeeze(1)  # Compressed channel dimension
        seg_labels = seg_labels.long()  # Convert to long
        
        # ============== Forward propagation ============= #
        if scaler:  # Mixed precision training
            with amp.autocast("cuda"):
                # -------- Get 6 outputs: seg_logits, logits_final, logits1, logits2, logits3, logits4 --------
                seg_logits, logits_final, logits1, logits2, logits3, logits4 = model(imgs)

                # 1) Principal classification loss
                loss_main = criterion_main(logits_final, class_labels)

                # 2) Split auxiliary task loss
                loss_seg = seg_criterion(seg_logits, seg_labels)

                # 3) Shallow auxiliary classification loss (including logits4)
                loss_aux1 = criterion_aux(logits1, class_labels)
                loss_aux2 = criterion_aux(logits2, class_labels)
                loss_aux3 = criterion_aux(logits3, class_labels)
                loss_aux4 = criterion_aux(logits4, class_labels)  # 新增
                loss_aux_total = loss_aux1 + loss_aux2 + loss_aux3 + loss_aux4

                # 4) Self-distillation loss: teacher =logits_final, student =[logits1, logits2, logits3, logits4]
                distill_loss = self_distillation_loss(
                    teacher_logits=logits_final,
                    student_logits_list=[logits1, logits2, logits3, logits4],
                    T=distill_T
                )

                # 5) Read from the model to define a hyperparameter
                alpha_aux = model.alpha_aux
                beta_distill = model.beta_distill
                alpha_seg = model.alpha_seg

                # 6) Total loss
                total_loss = (
                    loss_main
                    + alpha_seg * loss_seg
                    + alpha_aux * loss_aux_total
                    + beta_distill * distill_loss
                )

            # ============== Back propagation + gradient accumulation ============= #
            scaler.scale(total_loss / accumulation_steps).backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(data_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        else:
            # -------- Get 6 outputs: seg_logits, logits_final, logits1, logits2, logits3, logits4 --------
            seg_logits, logits_final, logits1, logits2, logits3, logits4 = model(imgs)

            # 1) Principal classification loss
            loss_main = criterion_main(logits_final, class_labels)
            # 2) Split auxiliary task loss
            loss_seg = seg_criterion(seg_logits, seg_labels)
            # 3) Shallow auxiliary classification loss (including logits4)
            loss_aux1 = criterion_aux(logits1, class_labels)
            loss_aux2 = criterion_aux(logits2, class_labels)
            loss_aux3 = criterion_aux(logits3, class_labels)
            loss_aux4 = criterion_aux(logits4, class_labels)
            loss_aux_total = loss_aux1 + loss_aux2 + loss_aux3 + loss_aux4

            # 4) Self-distillation loss
            distill_loss = self_distillation_loss(
                teacher_logits=logits_final,
                student_logits_list=[logits1, logits2, logits3, logits4],
                T=distill_T
            )

            # 5) Custom hyperparameter
            alpha_aux = model.alpha_aux
            beta_distill = model.beta_distill
            alpha_seg = model.alpha_seg

            # 6) Total loss
            total_loss = (
                loss_main
                + alpha_seg * loss_seg
                + alpha_aux * loss_aux_total
                + beta_distill * distill_loss
            )

            # ============== Back propagation + gradient accumulation ============= #
            (total_loss / accumulation_steps).backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(data_loader):
                optimizer.step()
                optimizer.zero_grad()

        # ============== Statistics and display ============= #
        _, preds = torch.max(logits_final, dim=1)
        running_corrects += torch.sum(preds == class_labels).item()
        total_samples += imgs.size(0)

        running_loss += loss_main.item() * imgs.size(0)  # Only main classification losses are shown
        current_loss = running_loss / total_samples
        current_acc = running_corrects / total_samples

        data_loader.set_description(
            f"Training Epoch {epoch} - Loss: {current_loss:.4f}, Acc: {current_acc:.4f}"
        )

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


# ------------------- 4) Evaluation function: Evaluate both classification and segmentation tasks ------------------- #
best_auc = 0.0

@torch.no_grad()
def evaluate(model, data_loader, device, epoch, num_classes=2):
    """
    Evaluate both classification and segmentation tasks:
    - Classification indicators: AUC, accuracy, F1 score, etc
    - Segmentation index: Dice coefficient (DSC), Jaccard Index (JI), Average surface distance (ASD), 95% Hausdorff distance (95HD)
    """
    global best_auc
    model.eval()

    # Initializes the category index related variables
    criterion_main = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_labels = []
    all_probs = []

    # Initializes the segmentation indicator correlation variables
    all_dsc = []
    all_iou = []
    all_asd = []
    all_hd95 = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    data_loader.set_description(f"Evaluating Epoch {epoch}")

    for batch_idx, batch in enumerate(data_loader):
        # Get images, category labels, and split labels
        imgs, class_labels, seg_labels = batch  # Modify here to get the split label
        imgs = imgs.to(device)
        class_labels = class_labels.to(device)
        seg_labels = seg_labels.to(device)
        seg_labels = seg_labels.squeeze(1)  # Compressed channel dimension
        seg_labels = seg_labels.long()

        # Forward reasoning
        outputs = model(imgs)
        seg_logits, logits_final, logits1, logits2, logits3, logits4 = outputs  #seg_logits([2, 2, 64, 64, 64])

        # ----------------- Computational classification index----------------- #
        loss_main = criterion_main(logits_final, class_labels)
        running_loss += loss_main.item() * imgs.size(0)

        _, preds = torch.max(logits_final, dim=1)
        running_corrects += torch.sum(preds == class_labels).item()
        total_samples += imgs.size(0)

        probs = F.softmax(logits_final, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(class_labels.detach().cpu().numpy())

        # ----------------- Computed segmentation index ----------------- #
        seg_preds = torch.argmax(seg_logits, dim=1)  # Gets the prediction category for each pixel
        seg_preds_np = seg_preds.detach().cpu().numpy().astype(np.uint8)
        seg_labels_np = seg_labels.detach().cpu().numpy().astype(np.uint8)

        # Calculate the segmentation index on a sample by sample basis
        for i in range(seg_preds_np.shape[0]):
            pred = seg_preds_np[i]
            label = seg_labels_np[i]

            # Dice coefficient and Jaccard index were calculated
            intersection = np.logical_and(pred, label)
            union = np.logical_or(pred, label)
            dice = (2.0 * intersection.sum()) / (pred.sum() + label.sum() + 1e-7)
            iou = intersection.sum() / (union.sum() + 1e-7)
            all_dsc.append(dice)
            all_iou.append(iou)

            # Calculate ASD and 95HD (handling exceptions)
            asd_val = assd(pred, label)
            hd_val = hd95(pred, label)
            if not np.isnan(asd_val):
                all_asd.append(asd_val)
            if not np.isnan(hd_val):
                all_hd95.append(hd_val)

        # Update progress bar shows classification loss and accuracy
        current_loss = running_loss / total_samples
        current_acc = running_corrects / total_samples
        data_loader.set_description(f"Evaluating Epoch {epoch} - Loss: {current_loss:.4f}, Acc: {current_acc:.4f}")

    # ----------------- Aggregate classification index ----------------- #
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    auc_value = roc_auc_score(all_labels, all_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    youden_j = tpr - fpr
    max_j_index = np.argmax(youden_j)
    cutoff = thresholds[max_j_index]
    new_preds = (np.array(all_probs) >= cutoff).astype(int)
    youden_acc = np.mean(new_preds == np.array(all_labels))
    cm = confusion_matrix(all_labels, new_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = precision_score(all_labels, new_preds, average='binary')
    recall = recall_score(all_labels, new_preds, average='binary')
    f1 = f1_score(all_labels, new_preds, average='binary')

    # ----------------- Summary segmentation index ----------------- #
    mean_dsc = np.mean(all_dsc) if all_dsc else 0.0
    mean_iou = np.mean(all_iou) if all_iou else 0.0
    mean_asd = np.mean(all_asd) if all_asd else 0.0
    mean_hd95 = np.mean(all_hd95) if all_hd95 else 0.0

    # ----------------- Print result ----------------- #
    print(f"\n==== Evaluation (Epoch {epoch}) ====")
    print("[Classification index]")
    print(f"ArgMax Loss       = {epoch_loss:.4f}")
    print(f"ArgMax Accuracy   = {epoch_acc:.4f}")
    print(f"AUC               = {auc_value:.4f}")
    print(f"Youden's J cutoff = {cutoff:.4f}")
    print(f"Acc(@Cutoff)      = {youden_acc:.4f}")
    print(f"Sensitivity       = {sensitivity:.4f}")
    print(f"Specificity       = {specificity:.4f}")
    print(f"Precision         = {precision:.4f}")
    print(f"F1 Score          = {f1:.4f}")

    print("\n[Segmentation index]")
    print(f"Dice coefficient (DSC)    = {mean_dsc:.4f}")
    print(f"Jaccard Index (IoU) = {mean_iou:.4f}")
    print(f"Average Surface Distance (ASD) = {mean_asd:.4f}")
    print(f"95% Hausdorff distance = {mean_hd95:.4f}")

    print("\nNormalized Confusion Matrix:")
    print(cm_normalized)
    print("\nClassification Report (Youden cutoff):")
    print(classification_report(all_labels, new_preds, digits=4, zero_division=0))

    # ----------------- Save the results and model ----------------- #
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    if auc_value > best_auc:
        print(f"Epoch {epoch} AUC ({auc_value:.4f}) > best AUC ({best_auc:.4f}), saving.")
        best_auc = auc_value

        # Save classification and segmentation indicator reports
        report_str = classification_report(all_labels, new_preds, digits=4, zero_division=0)
        with open(os.path.join(results_dir, "best_classification_report.txt"), "w") as f:
            f.write(f"Epoch: {epoch}\n")
            f.write(f"AUC: {auc_value:.4f}\n")
            f.write(f"Accuracy: {epoch_acc:.4f}\n")
            f.write(f"Sensitivity: {sensitivity:.4f}\n")
            f.write(f"Specificity: {specificity:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Dice coefficient (DSC): {mean_dsc:.4f}\n")
            f.write(f"Jaccard Index (IoU): {mean_iou:.4f}\n")
            f.write(f"Average Surface Distance (ASD): {mean_asd:.4f}\n")
            f.write(f"95% Hausdorff distance: {mean_hd95:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report_str)

        # Save confusion matrix and ROC curve (original code remains unchanged)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm_normalized, annot=True, fmt=".4f", cmap="Blues")
        plt.savefig(os.path.join(results_dir, "best_confusion_matrix.png"))
        plt.close()

        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, label=f'AUC={auc_value:.4f}')
        plt.scatter(fpr[max_j_index], tpr[max_j_index], c='red', label=f'Cutoff={cutoff:.4f}')
        plt.legend()
        plt.savefig(os.path.join(results_dir, "best_roc_curve.png"))
        plt.close()

    return epoch_loss, youden_acc, auc_value, sensitivity, specificity