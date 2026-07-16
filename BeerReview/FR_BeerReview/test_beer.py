import argparse
import os
import time
import csv
import torch
import numpy as np

from beer import BeerData, BeerAnnotation
from embedding import get_embeddings, get_glove_embedding
from torch.utils.data import DataLoader

from model import GenEncShareModel, GenEncNoShareModel
from validate_util import validate_share

# Create reverse word mapping
def create_idx2word(word2idx):
    idx2word = {idx: word for word, idx in word2idx.items()}
    idx2word[0] = '<UNK>'  # Unknown token
    return idx2word

def parse():
    parser = argparse.ArgumentParser(
        description="FR")

    # dataset parameters
    parser.add_argument('--data_dir',
                        type=str,
                        default='/home/dinesh/Documents/SXAI/FR_Final/data/beer',
                        help='Path of the dataset')
    parser.add_argument('--aspect',
                        type=int,
                        default=0,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--annotation_path',
                        type=str,
                        default='/home/dinesh/Documents/SXAI/FR_Final/data/beer/annotations.json',
                        help='Path to the annotation')
    parser.add_argument('--max_length',
                        type=int,
                        default=256,
                        help='Max sequence length [default: 256]')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Testing batch size [default: 32 for CPU]')
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=256,
                        help='Training batch size used for model selection [default: 256]')
    # pretrained embeddings
    parser.add_argument('--embedding_dir',
                        type=str,
                        default='/home/dinesh/Documents/SXAI/DAR_Final/data/hotel/embeddings',
                        help='Dir. of pretrained embeddings [default: None]')
    parser.add_argument('--embedding_name',
                        type=str,
                        default='glove.6B.100d.txt',
                        help='File name of pretrained embeddings [default: None]')

    # model parameters
    parser.add_argument('--cell_type',
                        type=str,
                        default="GRU",
                        help='Cell type: LSTM, GRU [default: GRU]')
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='RNN cell layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='Network Dropout')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=100,
                        help='Embedding dims [default: 100]')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=200,
                        help='RNN hidden dims [default: 100]')
    parser.add_argument('--num_class',
                        type=int,
                        default=2,
                        help='Number of predicted classes [default: 2]')

    # ckpt parameters
    parser.add_argument('--output_dir',
                        type=str,
                        default='./res',
                        help='Base dir of output files')

    # learning parameters
    parser.add_argument('--epochs',
                        type=int,
                        default=300,
                        help='Number of training epoch')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='compliment learning rate [default: 1e-4]')
    parser.add_argument('--sparsity_lambda',
                        type=float,
                        default=10.0,
                        help='Sparsity trade-off [default: 10.]')
    parser.add_argument('--continuity_lambda',
                        type=float,
                        default=10.0,
                        help='Continuity trade-off [default: 10.]')
    parser.add_argument(
        '--sparsity_percentage',
        type=float,
        default=0.15,
        help='Regularizer to control highlight percentage [default: .15]')
    parser.add_argument(
        '--cls_lambda',
        type=float,
        default=0.9,
        help='lambda for classification loss')
    parser.add_argument('--gpu',
                        type=str,
                        default='-1',
                        help='id(s) for CUDA_VISIBLE_DEVICES [default: -1 for CPU]')
    parser.add_argument('--share',
                        type=int,
                        default=1,
                        help='Share model (1) or not (0)')
    parser.add_argument('--tau',
                        type=int,
                        default=1,
                        help='')
    parser.add_argument('--taudecay',
                        type=int,
                        default=1,
                        help='Tau decay value')
    parser.add_argument('--model_base_dir',
                        type=str,
                        default='./trained_model/beer',
                        help='Base directory containing trained models')
    parser.add_argument('--output_dir_csv',
                        type=str,
                        default='./Metric_test_results',
                        help='Directory to save CSV output')
    args = parser.parse_args()
    return args


#####################
# set random seed
#####################
torch.manual_seed(12252018)

#####################
# parse arguments
#####################
args = parse()
args.tau = [args.tau]  # Fixed: Convert to list properly
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

######################
# device - CPU only
######################
device = torch.device("cpu")
print(f"Using device: {device}")

######################
# load embedding
######################
if args.embedding_name == 'review+wiki.filtered.200.txt.gz':
    pretrained_embedding, word2idx = get_embeddings(os.path.join(args.embedding_dir, args.embedding_name))
else:
    pretrained_embedding, word2idx = get_glove_embedding(os.path.join(args.embedding_dir, args.embedding_name))
args.vocab_size = len(word2idx)
args.pretrained_embedding = pretrained_embedding

# Create reverse mapping for tokens to words
idx2word = create_idx2word(word2idx)

######################
# load dataset
######################
dev_data = BeerData(args.data_dir, args.aspect, 'dev', word2idx)
annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)

# Use testing batch size
dev_loader = DataLoader(dev_data, batch_size=args.batch_size)
annotation_loader = DataLoader(annotation_data, batch_size=args.batch_size)

######################
# load model
######################
if args.share == 1:
    model = GenEncShareModel(args)
elif args.share == 0:
    model = GenEncNoShareModel(args)
else:
    print('please choose share of 0 or 1')
    exit()

# Construct model path with .pkl extension - Pattern: FR_model_share_{share}_sparsity_{sparsity_percentage}_lr_{lr}_conl_{continuity_lambda}_spl_{sparsity_lambda}_aspect_{aspect}.pkl
# model_filename = f"FR_model_share_{args.share}_sparsity_{args.sparsity_percentage}_lr_{args.lr}_conl_{args.continuity_lambda}_spl_{args.sparsity_lambda}_aspect_{args.aspect}.pkl"
# Updated pattern to match your actual model files
model_filename = f"FR_model_aspect_{args.aspect}_batch_size_{args.train_batch_size}_cls_lambda_{args.cls_lambda}_continuity_lambda_{args.continuity_lambda}_epochs_{args.epochs}_lr_{args.lr}_max_length_{args.max_length}_share_{args.share}_sparsity_lambda_{args.sparsity_lambda}_sparsity_percentage_{args.sparsity_percentage}_taudecay_{args.taudecay}.pth"

# Try multiple possible paths
possible_paths = [
    os.path.join(args.model_base_dir, model_filename),
    os.path.join('./trained_model/beer', model_filename),
    os.path.join('./trained_model', model_filename),
    os.path.join('.', model_filename),
]

model_path = None
for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        break

print(f"Training batch size: {args.train_batch_size}")
print(f"Testing batch size: {args.batch_size}")
print(f"Looking for model: {model_filename}")

# Load model with map_location for CPU
try:
    if model_path:
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if checkpoint is state_dict or full model
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("Model loaded successfully from state_dict")
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
            print("Model loaded successfully from state_dict")
        else:
            model = checkpoint
            print("Model loaded successfully as full model")
    else:
        raise FileNotFoundError(f"Model file not found. Tried paths: {possible_paths}")
        
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check if the model file exists and is compatible.")
    exit(1)

model.to(device)
model.eval()

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

# Initialize results list
results = []
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

# MICRO aggregation
agg = {}
# MACRO aggregation
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

print("\nTesting on annotation dataset and generating CSV...")

with torch.no_grad():
    for batch_idx, (inputs, masks, labels, annotations) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(device)
        
        # Forward pass
        rationales, logits = model(inputs, masks)
        
        # Get predictions
        logits_soft = torch.softmax(logits, dim=-1)
        _, preds = torch.max(logits_soft, axis=-1)
        
        # Process each sample in the batch
        batch_size = inputs.size(0)
        for i in range(batch_size):
            review_id = batch_idx * args.batch_size + i
            
            # Convert token IDs to words
            review_tokens = inputs[i].cpu().numpy()
            review_mask = masks[i].cpu().numpy()
            review_text = convert_to_words(review_tokens, idx2word, review_mask)
            
            # Get generated rationale (where z[:,:,1] == 1)
            gen_rationale_mask = rationales[i, :, 1].cpu().numpy()
            gen_rationale_text = get_rationale_words(review_tokens, gen_rationale_mask, idx2word, review_mask)
            gen_highlighted = highlight_rationale_in_text(review_tokens, gen_rationale_mask, idx2word, review_mask)
            
            # Get annotated rationale
            annotated_rationale_mask = annotations[i].cpu().numpy()
            annotated_rationale_text = get_rationale_words(review_tokens, annotated_rationale_mask, idx2word, review_mask)
            annotated_highlighted = highlight_rationale_in_text(review_tokens, annotated_rationale_mask, idx2word, review_mask)
            
            # ===== MULTI-THRESHOLD IoU METRICS =====

            # convert masks to spans
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

            # predicted mask
            pred_mask = gen_rationale_mask

            R = mask_to_spans(pred_mask)
            A = mask_to_spans(annotated_rationale_mask)

            metrics = {}

            for t in thresholds:

                # ===== ERASER =====
                TP_e = 0  # Fixed: removed unused 'used' variable

                # ===== TOKEN LEVEL =====
                pred_tokens = (pred_mask > 0).astype(int)
                gold_tokens = (annotated_rationale_mask > 0).astype(int)

                TP_to = np.sum((pred_tokens == 1) & (gold_tokens == 1))
                FP_to = np.sum((pred_tokens == 1) & (gold_tokens == 0))
                FN_to = np.sum((pred_tokens == 0) & (gold_tokens == 1))

                # ===== STOREK =====
                C = 0

                # ===== MODIFIED =====
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

                # store all metrics
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

                # to
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

# Create output directory
output_dir = args.output_dir_csv
os.makedirs(output_dir, exist_ok=True)

# Create CSV filename
csv_filename = f"Metric_FR_test_results_share_{args.share}_sparsity_{args.sparsity_percentage}_lr_{args.lr}_conl_{args.continuity_lambda}_spl_{args.sparsity_lambda}_aspect_{args.aspect}_trainbs{args.train_batch_size}.csv"
csv_path = os.path.join(output_dir, csv_filename)

# Save results to CSV
save_results_to_csv(results, csv_path)
print(f"\nResults saved to: {csv_path}")
print(f"Total reviews processed: {len(results)}")

# Calculate and print metrics
print("\nCalculating metrics...")
with torch.no_grad():
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels)) in enumerate(dev_loader):
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        rationales, logits = model(inputs, masks)  # Fixed: unpack both return values
        logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(logits, axis=-1)
        
        TP += ((pred == 1) & (labels == 1)).cpu().sum().item()
        TN += ((pred == 0) & (labels == 0)).cpu().sum().item()
        FN += ((pred == 0) & (labels == 1)).cpu().sum().item()
        FP += ((pred == 1) & (labels == 0)).cpu().sum().item()
    
    # Handle division by zero
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (recall + precision) if (recall + precision) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    
    print("Dev dataset metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1_score:.4f}")

# Also run validation on annotation dataset for sparsity metrics
print("\nCalculating annotation metrics...")
annotation_results = validate_share(model, annotation_loader, device)
print("Annotation dataset metrics:")
print(f"  Sparsity: {100 * annotation_results[0]:.2f}%")
print(f"  Precision: {100 * annotation_results[1]:.2f}%")
print(f"  Recall: {100 * annotation_results[2]:.2f}%")
print(f"  F1-Score: {100 * annotation_results[3]:.2f}%")

# Save summary statistics
results_filename = f"Metric_FR_test_summary_share_{args.share}_sparsity_{args.sparsity_percentage}_lr_{args.lr}_conl_{args.continuity_lambda}_spl_{args.sparsity_lambda}_aspect_{args.aspect}_trainbs{args.train_batch_size}.txt"
results_filepath = os.path.join(output_dir, results_filename)

with open(results_filepath, 'w') as f:
    f.write("==== FR Model Test Results Summary ====\n")
    f.write(f"Device: CPU\n")
    f.write(f"Model: FR (Share={args.share})\n")
    f.write(f"Aspect: {args.aspect}\n")
    f.write(f"Training Batch Size: {args.train_batch_size}\n")
    f.write(f"Testing Batch Size: {args.batch_size}\n")
    f.write(f"\n--- Hyperparameters ---\n")
    f.write(f"sparsity_percentage: {args.sparsity_percentage}\n")
    f.write(f"learning_rate: {args.lr}\n")
    f.write(f"continuity_lambda: {args.continuity_lambda}\n")
    f.write(f"sparsity_lambda: {args.sparsity_lambda}\n")
    f.write(f"cls_lambda: {args.cls_lambda}\n")
    f.write(f"epochs: {args.epochs}\n")
    f.write(f"max_length: {args.max_length}\n")
    f.write(f"taudecay: {args.taudecay}\n")
    f.write(f"\n--- Dev Dataset Results ---\n")
    f.write(f"accuracy: {accuracy:.4f}\n")
    f.write(f"precision: {precision:.4f}\n")
    f.write(f"recall: {recall:.4f}\n")
    f.write(f"f1_score: {f1_score:.4f}\n")
    f.write(f"\n--- Annotation Dataset Results ---\n")
    f.write(f"sparsity: {annotation_results[0]:.4f}\n")
    f.write(f"precision: {annotation_results[1]:.4f}\n")
    f.write(f"recall: {annotation_results[2]:.4f}\n")
    f.write(f"f1: {annotation_results[3]:.4f}\n")
    f.write(f"\n--- Output Files ---\n")
    f.write(f"csv_file: {csv_path}\n")
    f.write(f"total_samples: {len(results)}\n")
    f.write(f"model_used: {model_path}\n")
    
    # ===== SPAN IoU METRICS (Micro & Macro) =====
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

        # to
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

        # to
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

print(f"\nSummary saved to: {results_filepath}")