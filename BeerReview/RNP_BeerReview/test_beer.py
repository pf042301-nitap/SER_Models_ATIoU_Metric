import argparse
import os
import time
import csv
import torch
import numpy as np

from beer import BeerData, BeerAnnotation
from embedding import get_glove_embedding
from torch.utils.data import DataLoader

# Import both RNP model classes
try:
    from model import GenEncShareModel, GenEncNoShareModel, Teacher
    # Try importing RNP model if it exists separately
    try:
        from model import RNPModel
        RNP_MODEL_AVAILABLE = True
    except ImportError:
        RNP_MODEL_AVAILABLE = False
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
        description="RNP Model Testing")

    # dataset parameters
    parser.add_argument('--data_dir',
                        type=str,
                        default='/home/dinesh/Documents/SXAI/RNP_Final_All/data/beer',
                        help='Path of the dataset')
    parser.add_argument('--aspect',
                        type=int,
                        default=0,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--annotation_path',
                        type=str,
                        default='/home/dinesh/Documents/SXAI/RNP_Final_All/data/beer/annotations.json',
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
                        default='/home/dinesh/Documents/SXAI/RNP_Final_All/data/hotel/embeddings',
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
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='Number of training epochs [default: 200]')
    parser.add_argument('--share',
                        type=int,
                        default=0,
                        help='Share model (1) or not (0) for RNP')
    parser.add_argument('--model_base_dir',
                        type=str,
                        default='/home/dinesh/Documents/SXAI/RNP_Final_All/RNP/trained_model',
                        help='Base directory containing trained models')
    parser.add_argument('--output_dir_csv',
                        type=str,
                        default='./test_results',
                        help='Directory to save CSV output')
    parser.add_argument('--seed',
                        type=int,
                        default=12252018,
                        help='Random seed used in training [default: 12252018]')
    parser.add_argument('--save',
                        type=int,
                        default=1,
                        help='Save results flag')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='GPU device number')
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
dev_data = BeerData(args.data_dir, args.aspect, 'dev', word2idx)
annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)

# Use testing batch size
dev_loader = DataLoader(dev_data, batch_size=args.batch_size)
annotation_loader = DataLoader(annotation_data, batch_size=args.batch_size)

######################
# load RNP model - AUTO-DETECT VERSION
######################

def find_model_file(model_base_dir, aspect, share, sparsity_percentage, continuity_lambda, sparsity_lambda):
    """Automatically find the best matching model file"""
    
    # Possible model directories
    model_dirs = [
        os.path.join(model_base_dir, 'beer'),
        model_base_dir,
        './trained_model/beer',
        './trained_model',
        '/home/dinesh/Documents/SXAI/RNP_Final_All/RNP/trained_model/beer',
        '/home/dinesh/Documents/SXAI/RNP_Final_All/RNP/trained_model',
    ]
    
    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            continue
            
        print(f"\nChecking directory: {model_dir}")
        
        # Get all model files
        try:
            all_files = os.listdir(model_dir)
            model_files = [f for f in all_files if f.endswith('.pkl') and 'RNP_model' in f]
        except:
            continue
            
        if not model_files:
            print(f"  No model files found in {model_dir}")
            continue
            
        print(f"  Found {len(model_files)} model files")
        
        # Try to find exact match first
        for model_file in model_files:
            # Check if file matches all parameters
            if f'aspect_{aspect}' in model_file and f'share_{share}' in model_file:
                # Parse sparsity percentage from filename
                if f'sparsity_percentage_{sparsity_percentage}' in model_file:
                    return os.path.join(model_dir, model_file)
                # Try without trailing zeros
                sp_str = f"{sparsity_percentage:.3f}".rstrip('0').rstrip('.')
                if f'sparsity_percentage_{sp_str}' in model_file:
                    return os.path.join(model_dir, model_file)
        
        # If no exact match, find closest match
        best_match = None
        best_score = 0
        
        for model_file in model_files:
            score = 0
            if f'aspect_{aspect}' in model_file:
                score += 10
            if f'share_{share}' in model_file:
                score += 10
            if f'continuity_lambda_{continuity_lambda}' in model_file or f'continuity_lambda_{int(continuity_lambda)}' in model_file:
                score += 15
            if f'sparsity_lambda_{sparsity_lambda}' in model_file or f'sparsity_lambda_{int(sparsity_lambda)}' in model_file:
                score += 15
            if f'sparsity_percentage_{sparsity_percentage}' in model_file:
                score += 20
            else:
                # Check for close sparsity values
                for part in model_file.split('_'):
                    if 'sparsity_percentage' in part:
                        continue
                    try:
                        val = float(part)
                        if abs(val - sparsity_percentage) < 0.01:
                            score += 15
                    except:
                        pass
            
            if score > best_score:
                best_score = score
                best_match = os.path.join(model_dir, model_file)
        
        if best_match and best_score >= 30:  # Minimum confidence threshold
            print(f"  Best match (score: {best_score}): {os.path.basename(best_match)}")
            return best_match
    
    return None

# Find the model file
model_path = find_model_file(
    args.model_base_dir,
    args.aspect,
    args.share,
    args.sparsity_percentage,
    args.continuity_lambda,
    args.sparsity_lambda
)

if model_path is None:
    print(f"\n" + "="*60)
    print(f"ERROR: Could not find model file for:")
    print(f"  aspect={args.aspect}")
    print(f"  share={args.share}")
    print(f"  sparsity_percentage={args.sparsity_percentage}")
    print(f"  continuity_lambda={args.continuity_lambda}")
    print(f"  sparsity_lambda={args.sparsity_lambda}")
    print(f"="*60)
    
    # Search and list all available models
    print("\nSearching for all available models...")
    search_dirs = [
        '/home/dinesh/Documents/SXAI/RNP_Final_All/RNP/trained_model/beer',
        '/home/dinesh/Documents/SXAI/RNP_Final_All/RNP/trained_model',
        './trained_model/beer',
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"\nModels in {search_dir}:")
            for f in os.listdir(search_dir):
                if f.endswith('.pkl'):
                    print(f"  - {f}")
    exit(1)

# Initialize model
print(f"\n" + "="*60)
print(f"Loading model from: {model_path}")
print(f"="*60)

if args.share == 1:
    model = GenEncShareModel(args)
elif args.share == 0:
    model = GenEncNoShareModel(args)
else:
    print('Please choose share of 0 or 1')
    exit()

# Load model
try:
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try different loading strategies
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("Model loaded from state_dict")
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        print("Model loaded from state_dict")
    else:
        model = checkpoint
        print("Model loaded as full model")
    
    model.to(device)
    model.eval()
    print("Model successfully loaded and set to eval mode")
    
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

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

        for t in [0.1,0.2,0.3,0.4,0.5]:
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
        try:
            rationales, logits = model(inputs, masks)
            has_rationales = True
        except:
            try:
                logits = model(inputs, masks)
                has_rationales = False
                rationales = None
            except Exception as e:
                print(f"Forward pass error in batch {batch_idx}: {e}")
                continue
        
        # Get predictions
        logits_soft = torch.softmax(logits, dim=-1)
        _, preds = torch.max(logits_soft, axis=-1)
        
        # Process each sample
        batch_size = inputs.size(0)
        for i in range(batch_size):
            review_id = batch_idx * args.batch_size + i
            
            # Convert token IDs to words
            review_tokens = inputs[i].cpu().numpy()
            review_mask = masks[i].cpu().numpy()
            review_text = convert_to_words(review_tokens, idx2word, review_mask)
            
            # Get generated rationale
            if has_rationales and rationales is not None:
                try:
                    if len(rationales.shape) == 3:
                        gen_rationale_mask = rationales[i, :, 1].cpu().numpy()
                    else:
                        gen_rationale_mask = rationales[i].cpu().numpy()
                    gen_rationale_text = get_rationale_words(review_tokens, gen_rationale_mask, idx2word, review_mask)
                    gen_highlighted = highlight_rationale_in_text(review_tokens, gen_rationale_mask, idx2word, review_mask)
                except:
                    gen_rationale_text = "[Rationale extraction failed]"
                    gen_highlighted = "[Rationale extraction failed]"
            else:
                gen_rationale_text = "[No rationales generated]"
                gen_highlighted = "[No rationales generated]"
                gen_rationale_mask = np.zeros(len(review_mask))
            
            # Get annotated rationale
            annotated_rationale_mask = annotations[i].cpu().numpy()
            annotated_rationale_text = get_rationale_words(review_tokens, annotated_rationale_mask, idx2word, review_mask)
            annotated_highlighted = highlight_rationale_in_text(review_tokens, annotated_rationale_mask, idx2word, review_mask)
            
            # Convert masks to spans
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

            pred_mask = gen_rationale_mask
            R = mask_to_spans(pred_mask)
            A = mask_to_spans(annotated_rationale_mask)

            metrics = {}

            for t in thresholds:
                # Token level metrics
                pred_tokens = (pred_mask > 0).astype(int)
                gold_tokens = (annotated_rationale_mask > 0).astype(int)

                TP_to = np.sum((pred_tokens == 1) & (gold_tokens == 1))
                FP_to = np.sum((pred_tokens == 1) & (gold_tokens == 0))
                FN_to = np.sum((pred_tokens == 0) & (gold_tokens == 1))

                # Span-based metrics
                TP_e = 0
                C = 0
                TP_len = 0
                matched_len = set()

                for r in R:
                    best_iou = 0
                    best_idx = None
                    best_overlap = 0

                    for j, a in enumerate(A):
                        inter = len(set(r) & set(a))
                        union = len(set(r) | set(a))
                        iou = inter / union if union else 0

                        if iou > best_iou:
                            best_iou = iou
                            best_idx = j
                            best_overlap = inter

                    if best_iou >= t:
                        TP_e += 1
                        C += len(r)
                        
                        if best_idx not in matched_len:
                            TP_len += best_overlap
                            matched_len.add(best_idx)

                total_pred_len = sum(len(r) for r in R)
                total_ann_len = sum(len(a) for a in A)

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
                
                # Update aggregates
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

                # Macro scores for this sample
                # ERASER
                TP = metrics[f"t_{t}_eraser_tp"]
                FP = metrics[f"t_{t}_eraser_fp"]
                FN = metrics[f"t_{t}_eraser_fn"]
                p_e = TP/(TP+FP) if (TP+FP)>0 else 0
                r_e = TP/(TP+FN) if (TP+FN)>0 else 0
                f1_e = 2*p_e*r_e/(p_e+r_e) if (p_e+r_e)>0 else 0
                macro[t]['eraser_prec'].append(p_e)
                macro[t]['eraser_rec'].append(r_e)
                macro[t]['eraser_f1'].append(f1_e)

                # TOKEN
                TP = metrics[f"t_{t}_to_tp"]
                FP = metrics[f"t_{t}_to_fp"]
                FN = metrics[f"t_{t}_to_fn"]
                p_t = TP/(TP+FP) if (TP+FP)>0 else 0
                r_t = TP/(TP+FN) if (TP+FN)>0 else 0
                f1_t = 2*p_t*r_t/(p_t+r_t) if (p_t+r_t)>0 else 0
                macro[t]['to_prec'].append(p_t)
                macro[t]['to_rec'].append(r_t)
                macro[t]['to_f1'].append(f1_t)

                # STOREK
                p_s = metrics[f"t_{t}_storek_precision"]
                r_s = metrics[f"t_{t}_storek_recall"]
                f1_s = 2*p_s*r_s/(p_s+r_s) if (p_s+r_s)>0 else 0
                macro[t]['storek_prec'].append(p_s)
                macro[t]['storek_rec'].append(r_s)
                macro[t]['storek_f1'].append(f1_s)

                # MODIFIED
                TP = metrics[f"t_{t}_modified_tp_len"]
                FP = metrics[f"t_{t}_modified_fp_len"]
                FN = metrics[f"t_{t}_modified_fn_len"]
                p_m = TP/(TP+FP) if (TP+FP)>0 else 0
                r_m = TP/(TP+FN) if (TP+FN)>0 else 0
                f1_m = 2*p_m*r_m/(p_m+r_m) if (p_m+r_m)>0 else 0
                macro[t]['mod_prec'].append(p_m)
                macro[t]['mod_rec'].append(r_m)
                macro[t]['mod_f1'].append(f1_m)
            
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
csv_filename = f"Metric_RNP_test_results_aspect_{args.aspect}_share_{args.share}_sparsity_{args.sparsity_percentage}_conl_{args.continuity_lambda}_spl_{args.sparsity_lambda}.csv"
csv_path = os.path.join(output_dir, csv_filename)

# Save results to CSV
save_results_to_csv(results, csv_path)
print(f"\nResults saved to: {csv_path}")
print(f"Total reviews processed: {len(results)}")

# Calculate and print dev metrics
print("\nCalculating dev dataset metrics...")
with torch.no_grad():
    TP = TN = FN = FP = 0
    for batch, (inputs, masks, labels) in enumerate(dev_loader):
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        try:
            if has_rationales:
                _, logits = model(inputs, masks)
            else:
                logits = model(inputs, masks)
            logits_soft = torch.softmax(logits, dim=-1)
            _, pred = torch.max(logits_soft, axis=-1)
            TP += ((pred == 1) & (labels == 1)).cpu().sum().item()
            TN += ((pred == 0) & (labels == 0)).cpu().sum().item()
            FN += ((pred == 0) & (labels == 1)).cpu().sum().item()
            FP += ((pred == 1) & (labels == 0)).cpu().sum().item()
        except:
            continue
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (recall + precision) if (recall + precision) > 0 else 0
    
    print("Dev dataset metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1_score:.4f}")

# Save summary
results_filename = f"Metric_RNP_test_summary_aspect_{args.aspect}_share_{args.share}_sparsity_{args.sparsity_percentage}_conl_{args.continuity_lambda}_spl_{args.sparsity_lambda}.txt"
results_filepath = os.path.join(output_dir, results_filename)

with open(results_filepath, 'w') as f:
    f.write("==== RNP Model Test Results Summary ====\n")
    f.write(f"Model Path: {model_path}\n")
    f.write(f"Aspect: {args.aspect}\n")
    f.write(f"Share: {args.share}\n")
    f.write(f"Sparsity Percentage: {args.sparsity_percentage}\n")
    f.write(f"Continuity Lambda: {args.continuity_lambda}\n")
    f.write(f"Sparsity Lambda: {args.sparsity_lambda}\n")
    f.write(f"Learning Rate: {args.lr}\n")
    f.write(f"Epochs: {args.epochs}\n\n")
    
    f.write("--- Dev Dataset Results ---\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1_score:.4f}\n\n")
    
    f.write("--- Span IoU Metrics (Micro) ---\n")
    for t in thresholds:
        f.write(f"\nThreshold t = {t}:\n")
        # ERASER
        TP = agg[t]['eraser_TP']; FP = agg[t]['eraser_FP']; FN = agg[t]['eraser_FN']
        p = TP/(TP+FP) if (TP+FP)>0 else 0
        r = TP/(TP+FN) if (TP+FN)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        f.write(f"  ERASER - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")
        
        # TOKEN
        TP = agg[t]['to_TP']; FP = agg[t]['to_FP']; FN = agg[t]['to_FN']
        p = TP/(TP+FP) if (TP+FP)>0 else 0
        r = TP/(TP+FN) if (TP+FN)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        f.write(f"  TOKEN - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")
        
        # STOREK
        C = agg[t]['storek_C']; TR = agg[t]['storek_TR']; TA = agg[t]['storek_TA']
        p = C/TR if TR>0 else 0; r = C/TA if TA>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        f.write(f"  STOREK - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")
        
        # MODIFIED
        TP = agg[t]['mod_TP']; FP = agg[t]['mod_FP']; FN = agg[t]['mod_FN']
        p = TP/(TP+FP) if (TP+FP)>0 else 0
        r = TP/(TP+FN) if (TP+FN)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        f.write(f"  MODIFIED - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")
    
    f.write("\n--- Span IoU Metrics (Macro) ---\n")
    for t in thresholds:
        f.write(f"\nThreshold t = {t}:\n")
        f.write(f"  ERASER - P: {np.mean(macro[t]['eraser_prec']):.4f}, R: {np.mean(macro[t]['eraser_rec']):.4f}, F1: {np.mean(macro[t]['eraser_f1']):.4f}\n")
        f.write(f"  TOKEN - P: {np.mean(macro[t]['to_prec']):.4f}, R: {np.mean(macro[t]['to_rec']):.4f}, F1: {np.mean(macro[t]['to_f1']):.4f}\n")
        f.write(f"  STOREK - P: {np.mean(macro[t]['storek_prec']):.4f}, R: {np.mean(macro[t]['storek_rec']):.4f}, F1: {np.mean(macro[t]['storek_f1']):.4f}\n")
        f.write(f"  MODIFIED - P: {np.mean(macro[t]['mod_prec']):.4f}, R: {np.mean(macro[t]['mod_rec']):.4f}, F1: {np.mean(macro[t]['mod_f1']):.4f}\n")

print(f"\nSummary saved to: {results_filepath}")
print(f"\nDone! Check '{csv_path}' for complete results.")