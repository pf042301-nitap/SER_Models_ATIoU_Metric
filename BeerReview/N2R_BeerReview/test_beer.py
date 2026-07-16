import argparse
import os
import csv
import torch
import numpy as np

from beer import BeerAnnotation, BeerData
from embedding import get_glove_embedding
from torch.utils.data import DataLoader
from model import Mochangmodel


########################################
# PARSER
########################################
def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/beer')
    parser.add_argument('--aspect', type=int, default=0)
    parser.add_argument('--annotation_path', default='./data/beer/annotations.json')
    parser.add_argument('--max_length', type=int, default=256)

    parser.add_argument('--embedding_dir', default='./data/hotel/embeddings')
    parser.add_argument('--embedding_name', default='glove.6B.100d.txt')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=1)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--fr', type=int, default=0)
    parser.add_argument('--dr', type=int, default=0)
    parser.add_argument('--sp_norm', type=int, default=0)
    parser.add_argument('--dis_lr', type=float, default=0)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--sparsity_lambda', type=float, default=11.0)
    parser.add_argument('--continuity_lambda', type=float, default=12.0)
    parser.add_argument('--sparsity_percentage', type=float, default=0.08)

    parser.add_argument('--model_base_dir', default='./trained_model')
    parser.add_argument('--output_dir_csv', default='./Metric_test_mochang')

    return parser.parse_args()


########################################
# UTIL
########################################
def create_idx2word(word2idx):
    idx2word = {idx: word for word, idx in word2idx.items()}
    idx2word[0] = '<UNK>'
    return idx2word

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

def convert_to_words(token_ids, idx2word, mask):
    """Convert token IDs to words, respecting the mask"""
    words = []
    for i, token_id in enumerate(token_ids):
        if i >= len(mask):
            break
        if mask[i] == 1:
            words.append(idx2word.get(int(token_id), '<UNK>'))
    return " ".join(words)

def get_rationale_words(token_ids, rationale_mask, idx2word, original_mask):
    """Extract rationale words based on rationale mask"""
    words = []
    for i, token_id in enumerate(token_ids):
        if i >= len(original_mask):
            break
        if original_mask[i] == 1:
            if i < len(rationale_mask) and rationale_mask[i] == 1:
                words.append(idx2word.get(int(token_id), '<UNK>'))
            else:
                words.append("_")
    return " ".join(words)

def highlight_rationale_in_text(token_ids, rationale_mask, idx2word, original_mask):
    """Highlight rationale words in the original text with brackets"""
    words = []
    for i, token_id in enumerate(token_ids):
        if i >= len(original_mask):
            break
        if original_mask[i] == 1:
            word = idx2word.get(int(token_id), '<UNK>')
            if i < len(rationale_mask) and rationale_mask[i] == 1:
                words.append(f"[{word}]")
            else:
                words.append(word)
    return " ".join(words)

def save_results_to_csv(results, output_path):
    """Save results to CSV file"""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'review_id','review','generated_rationale','annotated_rationale',
            'generated_highlighted','annotated_highlighted',
            'pred_label','gt_label'
        ]

        for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
            fieldnames += [
                f't_{t}_eraser_tp', f't_{t}_eraser_fp', f't_{t}_eraser_fn',
                f't_{t}_to_tp', f't_{t}_to_fp', f't_{t}_to_fn',
                f't_{t}_storek_precision', f't_{t}_storek_recall',
                f't_{t}_modified_tp_len', f't_{t}_modified_fp_len', f't_{t}_modified_fn_len'
            ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)


########################################
# MAIN
########################################
args = parse()
device = torch.device("cpu")

# Load embeddings
embedding, word2idx = get_glove_embedding(
    os.path.join(args.embedding_dir, args.embedding_name)
)

args.vocab_size = len(word2idx)
args.pretrained_embedding = embedding

# Create reverse mapping
idx2word = create_idx2word(word2idx)

# Load data
dev_data = BeerData(args.data_dir, args.aspect, 'dev', word2idx)
annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)

dev_loader = DataLoader(dev_data, batch_size=args.batch_size)
annotation_loader = DataLoader(annotation_data, batch_size=args.batch_size)


########################################
# MODEL
########################################
model = Mochangmodel(args)

model_path = os.path.join(
    args.model_base_dir,
    'beer',
    f'N2R_model_share_0_sparsity_{args.sparsity_percentage}_lr_{args.lr}_conl_{args.continuity_lambda}_spl_{args.sparsity_lambda}_aspect_{args.aspect}.pkl'
)

print("Loading model:", model_path)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully")
except:
    model = torch.load(model_path, map_location=device)
    print("Model loaded as full model")

model.to(device)
model.eval()


########################################
# CLASSIFICATION METRICS
########################################
def compute_cls(loader):
    TP=TN=FP=FN=0

    with torch.no_grad():
        for batch in loader:
            if len(batch)==3:
                inputs, masks, labels = batch
            else:
                inputs, masks, labels, _ = batch

            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

            _, logits, _ = model.forward_predemb(inputs, masks)
            preds = torch.argmax(logits, dim=-1)

            TP += ((preds==1)&(labels==1)).sum().item()
            TN += ((preds==0)&(labels==0)).sum().item()
            FP += ((preds==1)&(labels==0)).sum().item()
            FN += ((preds==0)&(labels==1)).sum().item()

    prec = TP/(TP+FP+1e-9)
    rec = TP/(TP+FN+1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    acc = (TP+TN)/(TP+TN+FP+FN+1e-9)

    return acc, prec, rec, f1


dev_acc, dev_p, dev_r, dev_f1 = compute_cls(dev_loader)
ann_acc, ann_p, ann_r, ann_f1 = compute_cls(annotation_loader)


########################################
# METRICS INIT
########################################
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

agg = {}
macro = {}

for t in thresholds:
    agg[t] = {
        'eraser_TP': 0, 'eraser_FP': 0, 'eraser_FN': 0,
        'to_TP': 0, 'to_FP': 0, 'to_FN': 0,
        'storek_C': 0, 'storek_TR': 0, 'storek_TA': 0,
        'mod_TP': 0, 'mod_FP': 0, 'mod_FN': 0
    }

    macro[t] = {
        'eraser_prec': [], 'eraser_rec': [], 'eraser_f1': [],
        'to_prec': [], 'to_rec': [], 'to_f1': [],
        'storek_prec': [], 'storek_rec': [], 'storek_f1': [],
        'mod_prec': [], 'mod_rec': [], 'mod_f1': []
    }


########################################
# RESULTS LIST FOR CSV
########################################
results = []

########################################
# MAIN LOOP
########################################
print("\nTesting on annotation dataset and generating CSV...")

with torch.no_grad():
    for batch_idx, (inputs, masks, labels, annotations) in enumerate(annotation_loader):

        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(device)

        z, logits, _ = model.forward_predemb(inputs, masks)
        
        # Get predictions
        logits_soft = torch.softmax(logits, dim=-1)
        _, preds = torch.max(logits_soft, axis=-1)

        for i in range(inputs.size(0)):
            review_id = batch_idx * args.batch_size + i

            # Convert token IDs to words
            review_tokens = inputs[i].cpu().numpy()
            review_mask = masks[i].cpu().numpy()
            review_text = convert_to_words(review_tokens, idx2word, review_mask)

            # Get generated rationale
            gen_rationale_mask = z[i, :, 1].cpu().numpy()
            gen_rationale_text = get_rationale_words(review_tokens, gen_rationale_mask, idx2word, review_mask)
            gen_highlighted = highlight_rationale_in_text(review_tokens, gen_rationale_mask, idx2word, review_mask)

            # Get annotated rationale
            annotated_rationale_mask = annotations[i].cpu().numpy()
            annotated_rationale_text = get_rationale_words(review_tokens, annotated_rationale_mask, idx2word, review_mask)
            annotated_highlighted = highlight_rationale_in_text(review_tokens, annotated_rationale_mask, idx2word, review_mask)

            # Convert masks to spans
            R = mask_to_spans(gen_rationale_mask)
            A = mask_to_spans(annotated_rationale_mask)

            # Token level metrics
            pred_tokens = (gen_rationale_mask > 0).astype(int)
            gold_tokens = (annotated_rationale_mask > 0).astype(int)

            TP_to = np.sum((pred_tokens == 1) & (gold_tokens == 1))
            FP_to = np.sum((pred_tokens == 1) & (gold_tokens == 0))
            FN_to = np.sum((pred_tokens == 0) & (gold_tokens == 1))

            metrics = {}

            for t in thresholds:

                # ===== ERASER & MODIFIED & STOREK =====
                TP_e, used = 0, set()
                C = 0
                TP_len, matched_len = 0, set()

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

                    f"t_{t}_to_tp": int(TP_to),
                    f"t_{t}_to_fp": int(FP_to),
                    f"t_{t}_to_fn": int(FN_to),

                    f"t_{t}_storek_precision": C / total_pred_len if total_pred_len else 0,
                    f"t_{t}_storek_recall": C / total_ann_len if total_ann_len else 0,

                    f"t_{t}_modified_tp_len": TP_len,
                    f"t_{t}_modified_fp_len": total_pred_len - TP_len,
                    f"t_{t}_modified_fn_len": total_ann_len - TP_len
                })

                # ---- MICRO AGGREGATION ----
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

                # ---- MACRO AGGREGATION ----
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

            # Get labels
            pred_label = int(preds[i].cpu().item())
            gt_label = int(labels[i].cpu().item())

            row = {
                'review_id': review_id,
                'review': review_text,
                'generated_rationale': gen_rationale_text,
                'annotated_rationale': annotated_rationale_text,
                'generated_highlighted': gen_highlighted,
                'annotated_highlighted': annotated_highlighted,
                'pred_label': pred_label,
                'gt_label': gt_label
            }

            row.update(metrics)
            results.append(row)


########################################
# SAVE CSV
########################################
os.makedirs(args.output_dir_csv, exist_ok=True)

csv_filename = f"Metric_N2R_test_results_share_0_sparsity_{args.sparsity_percentage}_lr_{args.lr}_conl_{args.continuity_lambda}_spl_{args.sparsity_lambda}_aspect_{args.aspect}.csv"
csv_path = os.path.join(args.output_dir_csv, csv_filename)

save_results_to_csv(results, csv_path)
print(f"\nResults saved to: {csv_path}")
print(f"Total reviews processed: {len(results)}")


########################################
# PRINT CLASSIFICATION RESULTS
########################################
print("\nDev dataset metrics:")
print(f"  Accuracy: {dev_acc:.4f}")
print(f"  Precision: {dev_p:.4f}")
print(f"  Recall: {dev_r:.4f}")
print(f"  F1-Score: {dev_f1:.4f}")

print("\nAnnotation dataset metrics:")
print(f"  Accuracy: {ann_acc:.4f}")
print(f"  Precision: {ann_p:.4f}")
print(f"  Recall: {ann_r:.4f}")
print(f"  F1-Score: {ann_f1:.4f}")


########################################
# SAVE TXT SUMMARY
########################################
txt_path = os.path.join(args.output_dir_csv, csv_filename.replace('.csv', '.txt'))

with open(txt_path, "w") as f:

    f.write("==== N2R Model Test Results Summary ====\n")
    f.write(f"Device: {device}\n")
    f.write("Model: N2R (Share=0)\n")
    f.write(f"Aspect: {args.aspect}\n")
    f.write(f"Testing Batch Size: {args.batch_size}\n")

    f.write("\n--- Hyperparameters ---\n")
    f.write(f"sparsity_percentage: {args.sparsity_percentage}\n")
    f.write(f"learning_rate: {args.lr}\n")
    f.write(f"continuity_lambda: {args.continuity_lambda}\n")
    f.write(f"sparsity_lambda: {args.sparsity_lambda}\n")
    f.write(f"max_length: {args.max_length}\n")

    f.write("\n--- Dev Dataset Results ---\n")
    f.write(f"accuracy: {dev_acc:.4f}\n")
    f.write(f"precision: {dev_p:.4f}\n")
    f.write(f"recall: {dev_r:.4f}\n")
    f.write(f"f1_score: {dev_f1:.4f}\n")

    f.write("\n--- Annotation Dataset Results ---\n")
    f.write(f"accuracy: {ann_acc:.4f}\n")
    f.write(f"precision: {ann_p:.4f}\n")
    f.write(f"recall: {ann_r:.4f}\n")
    f.write(f"f1_score: {ann_f1:.4f}\n")

    f.write("\n--- Output Files ---\n")
    f.write(f"csv_file: {csv_path}\n")
    f.write(f"total_samples: {len(results)}\n")
    f.write(f"model_used: {model_path}\n")

    f.write("\n--- Span IoU Metrics (Micro & Macro) ---\n")

    for t in thresholds:
        f.write(f"\nThreshold t = {t}\n")

        # ===== MICRO =====
        f.write("Micro:\n")

        # ERASER
        TP = agg[t]['eraser_TP']
        FP = agg[t]['eraser_FP']
        FN = agg[t]['eraser_FN']
        p = TP/(TP+FP) if (TP+FP)>0 else 0
        r = TP/(TP+FN) if (TP+FN)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        f.write(f"  ERASER -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")

        # TO (token level)
        TP = agg[t]['to_TP']
        FP = agg[t]['to_FP']
        FN = agg[t]['to_FN']
        p = TP/(TP+FP) if (TP+FP)>0 else 0
        r = TP/(TP+FN) if (TP+FN)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        f.write(f"  TO -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")

        # Storek
        C = agg[t]['storek_C']
        TR = agg[t]['storek_TR']
        TA = agg[t]['storek_TA']
        p = C/TR if TR>0 else 0
        r = C/TA if TA>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        f.write(f"  Storek -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")

        # Modified
        TP = agg[t]['mod_TP']
        FP = agg[t]['mod_FP']
        FN = agg[t]['mod_FN']
        p = TP/(TP+FP) if (TP+FP)>0 else 0
        r = TP/(TP+FN) if (TP+FN)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        f.write(f"  Modified -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")

        # ===== MACRO =====
        f.write("Macro:\n")

        # ERASER
        p = np.mean(macro[t]['eraser_prec']) if macro[t]['eraser_prec'] else 0
        r = np.mean(macro[t]['eraser_rec']) if macro[t]['eraser_rec'] else 0
        f1 = np.mean(macro[t]['eraser_f1']) if macro[t]['eraser_f1'] else 0
        f.write(f"  ERASER -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")

        # TO
        p = np.mean(macro[t]['to_prec']) if macro[t]['to_prec'] else 0
        r = np.mean(macro[t]['to_rec']) if macro[t]['to_rec'] else 0
        f1 = np.mean(macro[t]['to_f1']) if macro[t]['to_f1'] else 0
        f.write(f"  TO -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")

        # Storek
        p = np.mean(macro[t]['storek_prec']) if macro[t]['storek_prec'] else 0
        r = np.mean(macro[t]['storek_rec']) if macro[t]['storek_rec'] else 0
        f1 = np.mean(macro[t]['storek_f1']) if macro[t]['storek_f1'] else 0
        f.write(f"  Storek -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")

        # Modified
        p = np.mean(macro[t]['mod_prec']) if macro[t]['mod_prec'] else 0
        r = np.mean(macro[t]['mod_rec']) if macro[t]['mod_rec'] else 0
        f1 = np.mean(macro[t]['mod_f1']) if macro[t]['mod_f1'] else 0
        f.write(f"  Modified -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")

print(f"\nSummary saved to: {txt_path}")
print("\nDONE ✅")