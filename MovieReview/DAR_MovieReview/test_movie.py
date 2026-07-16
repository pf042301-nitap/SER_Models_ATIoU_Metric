import argparse
import os
import time
import csv
import torch
import numpy as np


# from beer import BeerData, BeerAnnotation
from movies import MovieData, MovieAnnotation
from embedding import get_glove_embedding
from torch.utils.data import DataLoader

# Import both DAR model classes
try:
    from model import GenEncShareModel, GenEncNoShareModel, Teacher
    # Try importing DAR model if it exists separately
    try:
        from model import DARModel
        DAR_MODEL_AVAILABLE = True
    except ImportError:
        DAR_MODEL_AVAILABLE = False
except ImportError as e:
    print(f"Error importing models: {e}")
    exit(1)

# Create reverse word mapping
def create_idx2word(word2idx):
    idx2word = {idx: word for word, idx in word2idx.items()}
    idx2word[0] = '<UNK>'  # Unknown token
    return idx2word

def parse():
    parser = argparse.ArgumentParser(
        description="DAR Model Testing")

    # dataset parameters
    parser.add_argument('--data_dir',
                        type=str,
                        default='/media/dinesh/4TBHDD/Misc_Downloads_25102025/Programs_Documents/noise_injection_crf/data_preprocessing/usr_movie_review/original',
                        help='Path of the dataset')
    # parser.add_argument('--aspect',
    #                     type=int,
    #                     default=0,
    #                     help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--annotation_path',
                        type=str,
                        default='/media/dinesh/4TBHDD/Misc_Downloads_25102025/Programs_Documents/noise_injection_crf/data_preprocessing/usr_movie_review/original/test.jsonl',
                        help='Path to the annotation')
    parser.add_argument('--max_length',
                        type=int,
                        default=256,
                        help='Max sequence length [default: 256]')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,  # Testing batch size
                        help='Testing batch size [default: 32 for CPU]')
    # pretrained embeddings
    parser.add_argument('--embedding_dir',
                        type=str,
                        default='./data/embeddings',
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

    # learning parameters
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='learning rate [default: 1e-4]')
    parser.add_argument('--sparsity_lambda',
                        type=float,
                        default=11.0,
                        help='Sparsity trade-off [default: 11.]')
    parser.add_argument('--continuity_lambda',
                        type=float,
                        default=12.0,
                        help='Continuity trade-off [default: 12.]')
    parser.add_argument(
        '--sparsity_percentage',
        type=float,
        default=0.08,
        help='Regularizer to control highlight percentage [default: .08]')
    parser.add_argument(
        '--cls_lambda',
        type=float,
        default=0.9,
        help='lambda for classification loss')
    parser.add_argument('--share',
                        type=int,
                        default=0,
                        help='Share model (1) or not (0) for DAR')
    parser.add_argument('--model_base_dir',
                        type=str,
                        default='/media/dinesh/4TBHDD/Misc_Downloads_25102025/Programs_Documents/SER_Metric/DAR_Movie_Final/save_model/movie',
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
pretrained_embedding, word2idx = get_glove_embedding(os.path.join(args.embedding_dir, args.embedding_name))
args.vocab_size = len(word2idx)
args.pretrained_embedding = pretrained_embedding

# Create reverse mapping for tokens to words
idx2word = create_idx2word(word2idx)

######################
# load dataset
######################
# dev_data = BeerData(args.data_dir, args.aspect, 'dev', word2idx)
# annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)

dev_data = MovieData(
    args.data_dir,
    'dev',
    word2idx,
    max_length=args.max_length
)

annotation_data = MovieAnnotation(
    args.annotation_path,
    word2idx,
    max_length=args.max_length
)

# Use testing batch size
dev_loader = DataLoader(dev_data, batch_size=args.batch_size)
annotation_loader = DataLoader(annotation_data, batch_size=args.batch_size)

######################
# load DAR model
######################
# Based on the error, the model has both gen and cls components
# So it's likely a GenEncNoShareModel when share=0
if args.share == 1:
    model = GenEncShareModel(args)
elif args.share == 0:
    model = GenEncNoShareModel(args)
else:
    print('Please choose share of 0 or 1')
    exit()

# Construct model path based on your exact format
########################################################
# Movie model path
########################################################

# model_path = os.path.join(
#     './trained_model/movie',
#     (
#         f'DAR_model_'
#         f'maxlen_{args.max_length}_'
#         f'share_{args.share}_'
#         f'sparsity_{args.sparsity_percentage:.2f}_'
#         f'lr_{args.lr}_'
#         f'conl_{args.continuity_lambda:.1f}_'
#         f'spl_{args.sparsity_lambda:.1f}.pth'
#     )
# )

model_path = os.path.join(
    './trained_model/movie',
    (
        f'DAR_model_'
        f'maxlen_{args.max_length}_'
        f'share_{args.share}_'
        f'sparsity_{args.sparsity_percentage}_'
        f'lr_{args.lr}_'
        f'conl_{args.continuity_lambda:.1f}_'
        f'spl_{args.sparsity_lambda:.1f}.pth'
    )
)

print(f"\nLoading DAR model from:\n{model_path}")
print(f"Testing batch size: {args.batch_size}")
print(f"Loading DAR model from: {model_path}")

# Load model with map_location for CPU
# Load model with map_location for CPU
try:
    if os.path.exists(model_path):

        print("Attempting to load model as state_dict...")

        try:
            model.load_state_dict(
                torch.load(model_path, map_location=device)
            )

            print("DAR model loaded successfully from state_dict")

        except Exception as e:

            print(f"Error loading as state_dict: {e}")
            print("Attempting to load as full model...")

            model = torch.load(
                model_path,
                map_location=device
            )

            print("DAR model loaded successfully as full model")

    else:
        raise FileNotFoundError(
            f"\nMovie model not found:\n{model_path}"
        )

except Exception as e:

    print(f"Error loading model: {e}")
    print("Please check if the model file exists and is compatible.")
    exit(1)

# Check if model is already loaded (not state_dict)
if isinstance(model, torch.nn.Module):
    print("Model is already a torch.nn.Module instance")
    model.to(device)
    model.eval()
else:
    print("ERROR: Model is not a torch.nn.Module instance")
    print(f"Model type: {type(model)}")
    exit(1)

def convert_to_words(token_ids, idx2word, mask):
    """Convert token IDs to words, respecting the mask"""
    words = []
    for i, token_id in enumerate(token_ids):
        if i >= len(mask):  # Safety check
            break
        if mask[i] == 1:  # Only include non-padding tokens
            words.append(idx2word.get(int(token_id), '<UNK>'))
    return " ".join(words)

def get_rationale_words(token_ids, rationale_mask, idx2word, original_mask):
    """Extract rationale words based on rationale mask"""
    words = []
    for i, token_id in enumerate(token_ids):
        if i >= len(original_mask):  # Safety check
            break
        if original_mask[i] == 1:  # Only include non-padding tokens
            if i < len(rationale_mask) and rationale_mask[i] == 1:
                words.append(idx2word.get(int(token_id), '<UNK>'))
            else:
                words.append("_")  # Placeholder for non-rationale words
    return " ".join(words)

def highlight_rationale_in_text(token_ids, rationale_mask, idx2word, original_mask):
    """Highlight rationale words in the original text with brackets"""
    words = []
    for i, token_id in enumerate(token_ids):
        if i >= len(original_mask):  # Safety check
            break
        if original_mask[i] == 1:  # Only include non-padding tokens
            word = idx2word.get(int(token_id), '<UNK>')
            if i < len(rationale_mask) and rationale_mask[i] == 1:
                words.append(f"[{word}]")  # Highlight rationale words
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
        
        # Forward pass - DAR model returns rationales and logits
        # Try different forward pass methods
        try:
            # Try the standard GenEnc model forward pass
            rationales, logits = model(inputs, masks)
            has_rationales = True
        except Exception as e:
            print(f"Error with standard forward pass: {e}")
            print("Trying alternative forward pass...")
            try:
                # Try just getting logits
                logits = model(inputs, masks)
                has_rationales = False
                rationales = None
            except Exception as e2:
                print(f"Error with alternative forward pass: {e2}")
                print("Skipping this batch...")
                continue
        
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
            
            # Get generated rationale if available
            if has_rationales and rationales is not None:
                try:
                    # Assuming rationales shape: [batch, seq_len, 2] where [:, :, 1] is rationale mask
                    gen_rationale_mask = rationales[i, :, 1].cpu().numpy()
                    gen_rationale_text = get_rationale_words(review_tokens, gen_rationale_mask, idx2word, review_mask)
                    gen_highlighted = highlight_rationale_in_text(review_tokens, gen_rationale_mask, idx2word, review_mask)
                except:
                    gen_rationale_text = "[Rationale extraction failed]"
                    gen_highlighted = "[Rationale extraction failed]"
            else:
                gen_rationale_text = "[No rationales generated]"
                gen_highlighted = "[No rationales generated]"
            
            # Get annotated rationale
            annotated_rationale_mask = annotations[i].cpu().numpy()
            annotated_rationale_text = get_rationale_words(review_tokens, annotated_rationale_mask, idx2word, review_mask)
            annotated_highlighted = highlight_rationale_in_text(review_tokens, annotated_rationale_mask, idx2word, review_mask)
            # ===== MULTI-THRESHOLD IoU METRICS =====


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
            if has_rationales and rationales is not None:
                pred_mask = gen_rationale_mask
            else:
                pred_mask = np.zeros_like(annotated_rationale_mask)

            R = mask_to_spans(pred_mask)
            A = mask_to_spans(annotated_rationale_mask)

            metrics = {}

            for t in thresholds:

                # ===== ERASER =====
                TP_e = 0

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

# Create CSV filename with the specified format
# csv_filename = f"Metric_DAR_test_results_share_{args.share}_sparsity_{args.sparsity_percentage}_lr_{args.lr}_conl_{args.continuity_lambda}_spl_{args.sparsity_lambda}_aspect_{args.aspect}.csv"
csv_filename = (
    f"Metric_DAR_movie_results_"
    f"maxlen_{args.max_length}_"
    f"share_{args.share}_"
    f"sparsity_{args.sparsity_percentage:.2f}_"
    f"lr_{args.lr}_"
    f"conl_{args.continuity_lambda:.1f}_"
    f"spl_{args.sparsity_lambda:.1f}.csv"
)
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
        
        try:
            if has_rationales:
                _, logits = model(inputs, masks)
            else:
                logits = model(inputs, masks)
        except:
            # If forward pass fails, skip this batch
            continue
            
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

# Annotation dataset metrics
print("\nCalculating annotation dataset metrics...")
with torch.no_grad():
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels, _)) in enumerate(annotation_loader):
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        
        try:
            if has_rationales:
                _, logits = model(inputs, masks)
            else:
                logits = model(inputs, masks)
        except:
            continue
            
        logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(logits, axis=-1)
        
        TP += ((pred == 1) & (labels == 1)).cpu().sum().item()
        TN += ((pred == 0) & (labels == 0)).cpu().sum().item()
        FN += ((pred == 0) & (labels == 1)).cpu().sum().item()
        FP += ((pred == 1) & (labels == 0)).cpu().sum().item()
    
    # Handle division by zero
    ann_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    ann_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    ann_f1_score = 2 * ann_precision * ann_recall / (ann_recall + ann_precision) if (ann_recall + ann_precision) > 0 else 0
    ann_accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    
    print("Annotation dataset metrics:")
    print(f"  Accuracy: {ann_accuracy:.4f}")
    print(f"  Precision: {ann_precision:.4f}")
    print(f"  Recall: {ann_recall:.4f}")
    print(f"  F1-Score: {ann_f1_score:.4f}")

# Save summary statistics
os.makedirs(output_dir, exist_ok=True)
# results_filename = f"Metric_DAR_test_summary_share_{args.share}_sparsity_{args.sparsity_percentage}_lr_{args.lr}_conl_{args.continuity_lambda}_spl_{args.sparsity_lambda}_aspect_{args.aspect}.txt"
results_filename = (
    f"Metric_DAR_movie_summary_"
    f"maxlen_{args.max_length}_"
    f"share_{args.share}_"
    f"sparsity_{args.sparsity_percentage:.2f}_"
    f"lr_{args.lr}_"
    f"conl_{args.continuity_lambda:.1f}_"
    f"spl_{args.sparsity_lambda:.1f}.txt"
)
results_filepath = os.path.join(output_dir, results_filename)

with open(results_filepath, 'w') as f:
    f.write("==== DAR Model Test Results Summary ====\n")
    f.write(f"Device: CPU\n")
    f.write(f"Model: RNP (Share={args.share})\n")
    f.write(f"Testing Batch Size: {args.batch_size}\n")
    f.write(f"\n--- Hyperparameters ---\n")
    f.write(f"sparsity_percentage: {args.sparsity_percentage}\n")
    f.write(f"learning_rate: {args.lr}\n")
    f.write(f"continuity_lambda: {args.continuity_lambda}\n")
    f.write(f"sparsity_lambda: {args.sparsity_lambda}\n")
    f.write(f"\n--- Dev Dataset Results ---\n")
    f.write(f"accuracy: {accuracy:.4f}\n")
    f.write(f"precision: {precision:.4f}\n")
    f.write(f"recall: {recall:.4f}\n")
    f.write(f"f1_score: {f1_score:.4f}\n")
    f.write(f"\n--- Annotation Dataset Results ---\n")
    f.write(f"accuracy: {ann_accuracy:.4f}\n")
    f.write(f"precision: {ann_precision:.4f}\n")
    f.write(f"recall: {ann_recall:.4f}\n")
    f.write(f"f1_score: {ann_f1_score:.4f}\n")
    f.write(f"\n--- Output Files ---\n")
    f.write(f"csv_file: {csv_path}\n")
    f.write(f"total_samples: {len(results)}\n")
    f.write(f"model_used: {model_path}\n")
    # ===== ADD THIS BLOCK =====
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