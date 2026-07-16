import torch
import torch.nn.functional as F

from metric import compute_micro_stats

def validate_share_mochang(model, annotation_loader, device, epoch):
    num_true_pos = 0.
    num_predicted_pos = 0.
    num_real_pos = 0.
    num_words = 0
    total_mochang = 0
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels, annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
            device)

        # rationales -- (batch_size, seq_length, 2)
        rationales, cls_logits, clsoutputs = model.forward_predemb(inputs, masks)

        num_true_pos_, num_predicted_pos_, num_real_pos_ = compute_micro_stats(
            annotations, rationales[:, :, 1])

        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        # TP predict and label both 1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict and label both 0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

        num_true_pos += num_true_pos_
        num_predicted_pos += num_predicted_pos_
        num_real_pos += num_real_pos_
        num_words += torch.sum(masks)

        mochang = torch.norm(clsoutputs, p=2, dim=1)
        total_mochang += torch.sum(mochang).item()

    micro_precision = num_true_pos / num_predicted_pos
    micro_recall = num_true_pos / num_real_pos
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    sparsity = num_predicted_pos / num_words

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    avg_mochang = total_mochang / (TP + TN + FP + FN)
    
    print(f"Epoch {epoch}: annotation dataset - recall:{recall:.4f}, precision:{precision:.4f}, f1-score:{f1_score:.4f}, accuracy:{accuracy:.4f}, mochang:{avg_mochang:.4f}")

    return sparsity, micro_precision, micro_recall, micro_f1, avg_mochang

def validate_annotation_sentence(model, annotation_loader, device):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels, annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
            device)

        # rationales -- (batch_size, seq_length, 2)
        cls_logits = model.train_one_step(inputs, masks)

        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        # TP predict and label both 1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict and label both 0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    print(f"annotation sentence dataset: recall:{recall:.4f}, precision:{precision:.4f}, f1-score:{f1_score:.4f}, accuracy:{accuracy:.4f}")

def validate_dev_sentence(model, dev_loader, device, epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    
    for (batch, (inputs, masks, labels)) in enumerate(dev_loader):
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales -- (batch_size, seq_length, 2)
        cls_logits = model.train_one_step(inputs, masks)

        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        # TP predict and label both 1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict and label both 0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f"Epoch {epoch}: dev dataset - recall:{recall:.4f}, precision:{precision:.4f}, f1-score:{f1_score:.4f}, accuracy:{accuracy:.4f}")
    
    return precision, recall, f1_score, accuracy

def validate_rationales(model, annotation_loader, device, epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    
    for (batch, (inputs, masks, labels, annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(
            device), annotations.to(device)

        masks = annotations
        logits = model.train_one_step(inputs, masks)

        soft_pred = F.softmax(logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        # TP predict and label both 1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict and label both 0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    print(f"Epoch {epoch}: rationale dataset - recall:{recall:.4f}, precision:{precision:.4f}, f1-score:{f1_score:.4f}, accuracy:{accuracy:.4f}")