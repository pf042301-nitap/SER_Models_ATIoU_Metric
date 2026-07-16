import torch
import torch.nn.functional as F

from metric import get_sparsity_loss, get_continuity_loss
import numpy as np

def train_pred_mochang(model, optimizer, dataset, device, args, epoch, grad, grad_loss):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len = len(dataset)
    
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        rationales, logits = model.forward(inputs, masks)

        # compute loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss
        
        # update gradient
        if args.dis_lr == 1:
            if sparsity == 0:
                lr_lambda = 1
            else:
                lr_lambda = sparsity
            if lr_lambda < 0.05:
                lr_lambda = 0.05
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * lr_lambda
        elif args.dis_lr == 0:
            pass
        else:
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] / args.dis_lr

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # mochang only updates generator
        name1 = []
        name2 = []

        for idx, p in model.cls.named_parameters():
            if p.requires_grad == True:
                name1.append(idx)
                p.requires_grad = False
        for idx, p in model.cls_fc.named_parameters():
            if p.requires_grad == True:
                name2.append(idx)
                p.requires_grad = False

        rationales, logits, last_hid_state = model.forward_predemb(inputs, masks)
        mochang = torch.norm(last_hid_state, p=2, dim=1)

        mochang_loss = -args.mochang_lambda * torch.mean(torch.log(1e-6 + mochang))

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = sparsity_loss + continuity_loss + mochang_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        n1 = 0
        n2 = 0
        for idx, p in model.cls.named_parameters():
            if idx in name1:
                p.requires_grad = True
                n1 += 1
        for idx, p in model.cls_fc.named_parameters():
            if idx in name2:
                p.requires_grad = True
                n2 += 1

        # compute training accuracy
        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict and label both 1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict and label both 0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    # Print statistics
    print(f'Epoch {epoch}: cls_loss={cls_l:.4f}, sparsity_loss={spar_l:.4f}, continuity_loss={cont_l:.4f}, train_sparsity={np.mean(train_sp):.4f}')
    
    return precision, recall, f1_score, accuracy

def train_pred_mochang_withoutacc(model, optimizer, dataset, device, args, epoch, grad, grad_loss):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len = len(dataset)
    
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        rationales = model.get_rationale(inputs, masks)
        logits = model.pred_with_rationale(inputs, masks, torch.detach(rationales))

        # compute loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss
        
        # update gradient
        if args.dis_lr == 1:
            if sparsity == 0:
                lr_lambda = 1
            else:
                lr_lambda = sparsity
            if lr_lambda < 0.05:
                lr_lambda = 0.05
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * lr_lambda
        elif args.dis_lr == 0:
            pass
        else:
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] / args.dis_lr

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # mochang only updates generator, freeze predictor parameters
        name1 = []
        name2 = []

        for idx, p in model.cls.named_parameters():
            if p.requires_grad == True:
                name1.append(idx)
                p.requires_grad = False
        for idx, p in model.cls_fc.named_parameters():
            if p.requires_grad == True:
                name2.append(idx)
                p.requires_grad = False

        rationales, logits, last_hid_state = model.forward_predemb(inputs, masks)
        mochang = torch.norm(last_hid_state, p=2, dim=1)

        mochang_loss = -args.mochang_lambda * torch.mean(torch.log(1e-6 + mochang))

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = sparsity_loss + continuity_loss + mochang_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        n1 = 0
        n2 = 0
        for idx, p in model.cls.named_parameters():
            if idx in name1:
                p.requires_grad = True
                n1 += 1
        for idx, p in model.cls_fc.named_parameters():
            if idx in name2:
                p.requires_grad = True
                n2 += 1

        # compute training accuracy
        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict and label both 1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict and label both 0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    # Print statistics
    print(f'Epoch {epoch}: cls_loss={cls_l:.4f}, sparsity_loss={spar_l:.4f}, continuity_loss={cont_l:.4f}, train_sparsity={np.mean(train_sp):.4f}')
    
    return precision, recall, f1_score, accuracy