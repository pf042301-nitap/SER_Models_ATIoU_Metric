# Updated `Metrics_test_guide.py`


import argparse
import os
import torch
import numpy as np

from model import GuidanceBasedRationaleModule
from train_utils import prepare_dataset, try_gpu


########################################
# PARSER
########################################
def parse():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--aspect', type=int, default=0)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', default='./Metric_test_guide')

    # dataset
    parser.add_argument('--dataset', type=str, default='movie')
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--dev_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)

    # model config
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--output_size', type=int, default=2)
    parser.add_argument('--cell_type', type=str, default='gru')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.0002)

    # rationale config
    parser.add_argument('--sparsity', type=float, default=0.06)
    parser.add_argument('--continuity_lambda', type=float, default=1.0)
    parser.add_argument('--sparsity_lambda', type=float, default=0.01)
    parser.add_argument('--guide_lambda', type=float, default=0.0)
    parser.add_argument('--match_lambda', type=float, default=0.0)
    parser.add_argument('--guide_decay', type=float, default=0.0)

    # misc
    parser.add_argument('--embeddings', type=str, default='data/embeddings/glove.6B.100d.txt')
    parser.add_argument('--task_type', type=str, default='classification')
    parser.add_argument('--fix_embedding', action='store_true', default=False)
    parser.add_argument('--model', type=str, default='sep')
    parser.add_argument('--gpu_id', type=int, default=0)

    return parser.parse_args()


########################################
# UTIL
########################################
def mask_to_spans(mask):
    spans, cur = [], []

    for i, v in enumerate(mask):
        if v == 1:
            cur.append(i)
        else:
            if cur:
                spans.append(tuple(cur))
                cur = []

    if cur:
        spans.append(tuple(cur))

    return spans


def compute_prf(tp, fp, fn):
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return p, r, f1


def safe_mean(x):
    return np.mean(x) if len(x) > 0 else 0


########################################
# CLASSIFICATION METRICS
########################################
def compute_cls(loader, model, device):

    TP = TN = FP = FN = 0

    with torch.no_grad():

        for batch in loader:

            x, mask, labels, _ = batch

            x = x.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            outputs = model(x, mask)

            logits = outputs.predictions

            preds = torch.argmax(logits, dim=-1)

            TP += ((preds == 1) & (labels == 1)).sum().item()
            TN += ((preds == 0) & (labels == 0)).sum().item()
            FP += ((preds == 1) & (labels == 0)).sum().item()
            FN += ((preds == 0) & (labels == 1)).sum().item()

    prec = TP / (TP + FP + 1e-9)
    rec = TP / (TP + FN + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-9)

    return acc, prec, rec, f1


########################################
# MAIN
########################################
args = parse()


device = try_gpu(args.gpu_id)

configs = {
    'aspect': args.aspect,
    'dataset': args.dataset,
    'embeddings': args.embeddings,
    'task_type': args.task_type,
    'train_path': args.train_path,
    'dev_path': args.dev_path,
    'test_path': args.test_path,
    'max_length': args.max_length,
    'batch_size': args.batch_size,
    'embedding_size': args.embedding_size,
    'hidden_size': args.hidden_size,
    'output_size': args.output_size,
    'cell_type': args.cell_type,
    'dropout': args.dropout,
    'sparsity': args.sparsity,
    'continuity_lambda': args.continuity_lambda,
    'sparsity_lambda': args.sparsity_lambda,
    'guide_lambda': args.guide_lambda,
    'match_lambda': args.match_lambda,
    'guide_decay': args.guide_decay,
    'fix_embedding': args.fix_embedding,
}


########################################
# DATA
########################################
print('Loading dataset...')

vocab, _, _, test_loader = prepare_dataset(configs)


########################################
# MODEL
########################################
print('Building model...')

model = GuidanceBasedRationaleModule(
    vocab_size=vocab.vocab_size,
    emb_size=configs['embedding_size'],
    hidden_size=configs['hidden_size'],
    output_size=configs['output_size'],
    dropout=configs['dropout'],
    sparsity=configs['sparsity'],
    continuity_lambda=configs['continuity_lambda'],
    sparsity_lambda=configs['sparsity_lambda'],
    cell_type=configs['cell_type'],
    pretrained_embedding=vocab.embedding_matrix,
    fix_embedding=configs['fix_embedding'],
    guide_lambda=configs['guide_lambda'],
    match_lambda=configs['match_lambda'],
    guide_decay=configs['guide_decay']
)

print('Loading model:', args.model_path)

model.load_state_dict(
    torch.load(args.model_path, map_location=device)
)

model.to(device)
model.eval()


########################################
# METRICS INIT
########################################
thresholds = [0.1,0.2,0.3,0.4,0.5]

agg = {}
macro = {}

for t in thresholds:
    agg[t] = {
        'eraser_TP':0,'eraser_FP':0,'eraser_FN':0,
        'to_TP':0,'to_FP':0,'to_FN':0,
        'storek_C':0,'storek_TR':0,'storek_TA':0,
        'mod_TP':0,'mod_FP':0,'mod_FN':0
    }

    macro[t] = {
        'eraser_prec':[], 'eraser_rec':[], 'eraser_f1':[],
        'to_prec':[], 'to_rec':[], 'to_f1':[],
        'storek_prec':[], 'storek_rec':[], 'storek_f1':[],
        'mod_prec':[], 'mod_rec':[], 'mod_f1':[]
    }



########################################
# EVALUATION LOOP
########################################
########################################
# EVALUATION LOOP
########################################
print('Running evaluation...')

with torch.no_grad():

    for batch in test_loader:

        x, mask, labels, ann = batch

        x = x.to(device)
        mask = mask.to(device)
        ann = ann.to(device)

        outputs = model(x, mask)

        z = outputs.rationales

        for i in range(x.size(0)):

            pred_mask = (z[i] > 0.5).int().cpu().numpy()
            ann_mask = ann[i].cpu().numpy()

            R = mask_to_spans(pred_mask)
            A = mask_to_spans(ann_mask)

            # Token-level TP, FP, FN
            pred_tokens = set(np.where(pred_mask == 1)[0])
            ann_tokens = set(np.where(ann_mask == 1)[0])
            
            TP_to = len(pred_tokens & ann_tokens)
            FP_to = len(pred_tokens - ann_tokens)
            FN_to = len(ann_tokens - pred_tokens)

            metrics = {}

            for t in thresholds:

                # ===== ERASER & MODIFIED & STOREK =====
                TP_e = 0
                C = 0
                TP_len = 0
                matched_len = set()

                for r in R:
                    best_iou, best_idx = 0, None
                    best_overlap = 0

                    for j, a in enumerate(A):
                        inter = len(set(r) & set(a))
                        union = len(set(r) | set(a))
                        iou = inter / union if union else 0

                        if iou > best_iou:
                            best_iou = iou
                            best_idx = j
                            best_overlap = inter

                    # ===== ERASER =====
                    if best_iou >= t:
                        TP_e += 1

                    # ===== STOREK =====
                    if best_iou >= t:
                        C += len(r)

                    # ===== MODIFIED =====
                    if best_iou >= t and best_idx not in matched_len:
                        TP_len += best_overlap
                        matched_len.add(best_idx)

                total_pred_len = sum(len(r) for r in R)
                total_ann_len = sum(len(a) for a in A)

                # Store metrics
                metrics.update({
                    f"t_{t}_eraser_tp": TP_e,
                    f"t_{t}_eraser_fp": len(R) - TP_e,
                    f"t_{t}_eraser_fn": len(A) - TP_e,

                    f"t_{t}_to_tp": TP_to,
                    f"t_{t}_to_fp": FP_to,
                    f"t_{t}_to_fn": FN_to,

                    f"t_{t}_storek_precision": C / total_pred_len if total_pred_len else 0,
                    f"t_{t}_storek_recall": C / total_ann_len if total_ann_len else 0,

                    f"t_{t}_modified_tp_len": TP_len,
                    f"t_{t}_modified_fp_len": total_pred_len - TP_len,
                    f"t_{t}_modified_fn_len": total_ann_len - TP_len
                })
            
                # ---- MICRO ----
                agg[t]['eraser_TP'] += metrics[f"t_{t}_eraser_tp"]
                agg[t]['eraser_FP'] += metrics[f"t_{t}_eraser_fp"]
                agg[t]['eraser_FN'] += metrics[f"t_{t}_eraser_fn"]

                agg[t]['to_TP'] += metrics[f"t_{t}_to_tp"]
                agg[t]['to_FP'] += metrics[f"t_{t}_to_fp"]
                agg[t]['to_FN'] += metrics[f"t_{t}_to_fn"]

                agg[t]['storek_C'] += metrics[f"t_{t}_storek_precision"] * total_pred_len
                agg[t]['storek_TR'] += total_pred_len
                agg[t]['storek_TA'] += total_ann_len

                agg[t]['mod_TP'] += metrics[f"t_{t}_modified_tp_len"]
                agg[t]['mod_FP'] += metrics[f"t_{t}_modified_fp_len"]
                agg[t]['mod_FN'] += metrics[f"t_{t}_modified_fn_len"]

                # ---- MACRO ----
                # ERASER
                TP = metrics[f"t_{t}_eraser_tp"]
                FP = metrics[f"t_{t}_eraser_fp"]
                FN = metrics[f"t_{t}_eraser_fn"]

                p_i = TP/(TP+FP) if (TP+FP)>0 else 0
                r_i = TP/(TP+FN) if (TP+FN)>0 else 0
                f1_i = 2*p_i*r_i/(p_i+r_i) if (p_i+r_i)>0 else 0

                macro[t]['eraser_prec'].append(p_i)
                macro[t]['eraser_rec'].append(r_i)
                macro[t]['eraser_f1'].append(f1_i)

                # TO (token level)
                TP = metrics[f"t_{t}_to_tp"]
                FP = metrics[f"t_{t}_to_fp"]
                FN = metrics[f"t_{t}_to_fn"]

                p_i = TP/(TP+FP) if (TP+FP)>0 else 0
                r_i = TP/(TP+FN) if (TP+FN)>0 else 0
                f1_i = 2*p_i*r_i/(p_i+r_i) if (p_i+r_i)>0 else 0

                macro[t]['to_prec'].append(p_i)
                macro[t]['to_rec'].append(r_i)
                macro[t]['to_f1'].append(f1_i)

                # STOREK
                p_i = metrics[f"t_{t}_storek_precision"]
                r_i = metrics[f"t_{t}_storek_recall"]
                f1_i = 2*p_i*r_i/(p_i+r_i) if (p_i+r_i)>0 else 0

                macro[t]['storek_prec'].append(p_i)
                macro[t]['storek_rec'].append(r_i)
                macro[t]['storek_f1'].append(f1_i)

                # MODIFIED
                TP = metrics[f"t_{t}_modified_tp_len"]
                FP = metrics[f"t_{t}_modified_fp_len"]
                FN = metrics[f"t_{t}_modified_fn_len"]

                p_i = TP/(TP+FP) if (TP+FP)>0 else 0
                r_i = TP/(TP+FN) if (TP+FN)>0 else 0
                f1_i = 2*p_i*r_i/(p_i+r_i) if (p_i+r_i)>0 else 0

                macro[t]['mod_prec'].append(p_i)
                macro[t]['mod_rec'].append(r_i)
                macro[t]['mod_f1'].append(f1_i)
########################################
# SAVE RESULTS
########################################
os.makedirs(args.output_dir, exist_ok=True)


txt_path = os.path.join(
    args.output_dir,
    os.path.basename(args.model_path).replace('.pt', '.txt')
)


dev_acc, dev_p, dev_r, dev_f1 = compute_cls(
    test_loader,
    model,
    device
)


with open(txt_path, 'w') as f:

    f.write('==== GRAT/GUIDE MODEL TEST RESULTS ====\n')

    f.write(f'Device: {device}\n')
    f.write(f'Dataset: {configs["dataset"]}\n')
    f.write(f'Model path: {args.model_path}\n')

    f.write('\n===== CONFIG =====\n')

    for k, v in configs.items():
        f.write(f'{k}: {v}\n')

    f.write('\n===== CLASSIFICATION =====\n')

    f.write(f'accuracy: {dev_acc:.4f}\n')
    f.write(f'precision: {dev_p:.4f}\n')
    f.write(f'recall: {dev_r:.4f}\n')
    f.write(f'f1_score: {dev_f1:.4f}\n')

    f.write('\n===== RATIONALE METRICS =====\n')

    f.write("\n--- Span IoU Metrics (Micro & Macro) ---\n")

    for t in thresholds:
        f.write(f"\nThreshold t = {t}\n")

        # ===== MICRO =====
        f.write("Micro:\n")

        # ERASER
        p,r,f1 = compute_prf(agg[t]['eraser_TP'], agg[t]['eraser_FP'], agg[t]['eraser_FN'])
        f.write(f"  ERASER -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")

        # TO (token level)
        p,r,f1 = compute_prf(agg[t]['to_TP'], agg[t]['to_FP'], agg[t]['to_FN'])
        f.write(f"  TO -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")

        # Storek
        p = agg[t]['storek_C']/agg[t]['storek_TR'] if agg[t]['storek_TR']>0 else 0
        r = agg[t]['storek_C']/agg[t]['storek_TA'] if agg[t]['storek_TA']>0 else 0
        f.write(f"  Storek -> P: {p:.4f}, R: {r:.4f}, F1: {2*p*r/(p+r+1e-9):.4f}\n")

        # Modified
        p,r,f1 = compute_prf(agg[t]['mod_TP'], agg[t]['mod_FP'], agg[t]['mod_FN'])
        f.write(f"  Modified -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")

        # ===== MACRO =====
        f.write("Macro:\n")

        # ERASER
        p = safe_mean(macro[t]['eraser_prec'])
        r = safe_mean(macro[t]['eraser_rec'])
        f1 = safe_mean(macro[t]['eraser_f1'])
        f.write(f"  ERASER -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")

        # TO
        p = safe_mean(macro[t]['to_prec'])
        r = safe_mean(macro[t]['to_rec'])
        f1 = safe_mean(macro[t]['to_f1'])
        f.write(f"  TO -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")

        # Storek
        p = safe_mean(macro[t]['storek_prec'])
        r = safe_mean(macro[t]['storek_rec'])
        f1 = safe_mean(macro[t]['storek_f1'])
        f.write(f"  Storek -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")

        # Modified
        p = safe_mean(macro[t]['mod_prec'])
        r = safe_mean(macro[t]['mod_rec'])
        f1 = safe_mean(macro[t]['mod_f1'])
        f.write(f"  Modified -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")

print("DONE ✅")
print("Saved:", txt_path)