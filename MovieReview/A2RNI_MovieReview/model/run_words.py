import json
import os
from argparse import ArgumentParser

import torch
import torch.optim as optim
from sklearn.metrics import classification_report
from termcolor import colored
from tqdm.auto import tqdm
from transformers import AutoTokenizer, logging

from models import (BlackBoxPredictor, RationaleExtractor,
                    RationaleExtractorFactory, RationalePredictor,
                    SelectorFactory)
from movies import DataLoaderFactory

logging.set_verbosity_error()

def parse_args():

    parser = ArgumentParser()

    # Whether to train and/or evaluate
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")

    # Whether to inject noise
    parser.add_argument("--inject_noise", action="store_true")

    # Magnitude of augmentation hyperparameter
    parser.add_argument('--noise_p', type=float, default=0.1)

    # Device
    parser.add_argument('--device', type=str, default='cuda')

    # Optimizer BB
    parser.add_argument('--bb_lr', type=float, default=2e-5)

    # Optimizer RP
    parser.add_argument('--rp_lr', type=float, default=2e-5)

    # Freeze BERT weights
    parser.add_argument('--freeze_encoder_bb', action="store_true")
    parser.add_argument('--freeze_encoder_rp', action="store_true")

    # Training
    parser.add_argument('--num_epochs', type=int, default=5)

    # Patience
    parser.add_argument('--patience', type=int, default=2)

    # Model proximity hyperparameter
    parser.add_argument('--proximity', type=float, default=0.1)

    # Model
    parser.add_argument('--save_path', type=str,
                        default=os.path.join("trained", "ours"))

    parser.add_argument('--model', type=str,
                        default='bert-base-uncased')

    parser.add_argument('--max_length', type=int, default=512)

    parser.add_argument('--batch_size', type=int, default=16)

    # Rationale Extraction hyperparameter
    parser.add_argument('--sparsity', type=float, default=0.2)

    # Dataset
    parser.add_argument('--data_path', type=str,
                        default=os.path.join("..", "..",
                                             "rnp_movie_review",
                                             "original"))

    # Selection method
    parser.add_argument('--selection_method',
                        choices=['words', 'span'],
                        default='words')

    # Eval-related
    parser.add_argument('--show_detail', action="store_true")

    args = parser.parse_args()

    # ======================================================
    # ONLY USE PASSED PARAMETERS FOR EXPERIMENT NAME
    # ======================================================

    exclude_params = {
        "dropout",
        "embedding_dim",
        "hidden_dim",
        "num_class",
        "num_layers",
        "gpu",
        "writer",
        "save",
        "pretrained_embedding"
    }

    passed_args = {
        arg.split("=")[0].replace("--", "")
        for arg in os.sys.argv[1:]
        if arg.startswith("--")
    }

    passed_args.discard("train")
    passed_args.discard("show_detail")

    numeric_params = {
        k: v for k, v in vars(args).items()
        if (
            k in passed_args
            and (
                isinstance(v, (int, float))
                or k == "selection_method"
            )
            and k not in exclude_params
        )
    }

    args.NI_name = "NI_" + "_".join(
        [f"{k}_{v}" for k, v in sorted(numeric_params.items())]
    )

    print("\nGenerated Experiment Name:")
    print(args.NI_name)

    return args

def main(args):
    if not args.train and not args.evaluate:
        print("Must append flag --train or --evaluate")
        return

    args.save_path = os.path.join(
        args.save_path,
        args.NI_name
    )

    print(f"\n[INFO] Saving to: {args.save_path}")

    # existing line
    checkpoint_dir = os.path.join(args.save_path, "checkpoints")


    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast = True)

    bb_model = BlackBoxPredictor(num_labels = 2, model = args.model, freeze_encoder = args.freeze_encoder_bb).to(args.device)
    print(f"Black Box Predictor: {get_num_params(bb_model)} parameters")

    rp_model = RationalePredictor(num_labels = 2, model = args.model, freeze_encoder = args.freeze_encoder_rp).to(args.device)
    print(f"Rationale Predictor: {get_num_params(rp_model)} parameters")

    rationale_selector = SelectorFactory(args.sparsity, args.max_length, tokenizer.pad_token_id, args.device).create_selector(args.selection_method)

    rationale_extractor = RationaleExtractorFactory(tokenizer, args.device, args.data_path).create_extractor(args.inject_noise)

    if args.train:
        os.makedirs(checkpoint_dir, exist_ok = True)
        train_loader = DataLoaderFactory(
            data_path = args.data_path,
            noise_p = args.noise_p,
            batch_size = args.batch_size,
            tokenizer = tokenizer,
            max_length = args.max_length,
            shuffle = True
        ).create_dataloader("train", args.inject_noise)
        valid_loader = DataLoaderFactory(
            data_path = args.data_path,
            noise_p = args.noise_p,
            batch_size = args.batch_size,
            tokenizer = tokenizer,
            max_length = args.max_length,
            shuffle = True
        ).create_dataloader("valid", args.inject_noise)

        bb_optimizer = optim.Adam(bb_model.parameters(), args.bb_lr)
        rp_optimizer = optim.Adam(rp_model.parameters(), args.rp_lr)

        validation_rationale_extractor = RationaleExtractor(tokenizer, args.device)

        train(
            bb_model = bb_model,
            bb_optimizer = bb_optimizer,
            rp_model = rp_model,
            rp_optimizer = rp_optimizer,
            train_loader = train_loader,
            valid_loader = valid_loader,
            eval_every = len(train_loader),
            device = args.device,
            proximity = args.proximity,
            num_epochs = args.num_epochs,
            patience = args.patience,
            checkpoint_dir = checkpoint_dir,
            rationale_selector = rationale_selector,
            rationale_extractor = rationale_extractor,
            validation_rationale_extractor = validation_rationale_extractor,
            selection_method = args.selection_method,
        )

    if args.evaluate:
        test_loader = DataLoaderFactory(
            data_path = args.data_path,
            noise_p = args.noise_p,
            batch_size = args.batch_size,
            tokenizer = tokenizer,
            max_length = args.max_length,
            shuffle = False
        ).create_dataloader("test", False)

        test_rationale_extractor = RationaleExtractor(tokenizer, args.device)

        evaluate(
            bb_model = bb_model,
            rp_model = rp_model,
            tokenizer = tokenizer,
            test_loader = test_loader,
            device = args.device,
            show_detail = args.show_detail,
            rationale_selector = rationale_selector,
            rationale_extractor = test_rationale_extractor,
            checkpoint_dir = checkpoint_dir,
            result_path = args.save_path,
            num_epochs = args.num_epochs,
            selection_method = args.selection_method,
        )

def train(
    bb_model,
    bb_optimizer,
    rp_model,
    rp_optimizer,
    train_loader,
    valid_loader,
    eval_every,
    device,
    proximity,
    num_epochs,
    patience,
    checkpoint_dir,
    rationale_selector,
    rationale_extractor,
    validation_rationale_extractor,
    selection_method,
    ):

    with tqdm(total=num_epochs * len(train_loader)) as pb:

        # Initialize statistics
        bb_running_train_loss = 0.0
        bb_best_train_loss = float("Inf")
        rp_running_train_loss = 0.0
        rp_best_train_loss = float("Inf")
        bb_running_valid_loss = 0.0
        bb_best_valid_loss = float("Inf")
        rp_running_valid_loss = 0.0
        rp_best_valid_loss = float("Inf")
        running_train_replace_ratio = 0.0
        running_valid_replace_ratio = 0.0
        global_step = 0
        metrics = []
        patience_left = patience

        # training loop
        bb_model.train()
        rp_model.train()
        for epoch in range(num_epochs):
            for batch in train_loader:

                if patience_left == 0:
                    pb.write("Patience is 0, early stopping")
                    break

                # generate prediction and token probs of being in a rationale
                batch.reviews_tokenized = batch.reviews_tokenized.to(device)
                att_pred, token_att = bb_model(**batch.reviews_tokenized)

                hard_mask = rationale_selector(
                    token_att = token_att,
                    input_ids = batch.reviews_tokenized.input_ids
                )
                rationale, _, replace_ratio = rationale_extractor(
                    batch = batch,
                    hard_mask = hard_mask
                )

                # predict from rationale
                hard_pred = rp_model(**rationale)
            
                bb_loss = bb_model.get_loss(
                    att_pred = att_pred,
                    hard_pred = hard_pred.detach(),
                    labels = batch.labels_bb.to(device),
                    proximity = proximity
                )

                rp_loss = rp_model.get_loss(
                    att_pred = att_pred.detach(),
                    hard_pred = hard_pred,
                    labels = batch.labels_rp.to(device),
                    proximity = proximity
                )

                bb_optimizer.zero_grad()
                rp_optimizer.zero_grad()

                bb_loss.backward()
                rp_loss.backward()

                bb_optimizer.step()
                rp_optimizer.step()
            
                pb.update(1)

                # update running values
                bb_running_train_loss += bb_loss.item()
                rp_running_train_loss += rp_loss.item()
                running_train_replace_ratio += replace_ratio
                global_step += 1

                # validation step
                if global_step % eval_every == 0:
                    bb_model.eval()
                    rp_model.eval()
                    with torch.no_grad():                    
                        for batch in valid_loader:
                            # generate prediction and token probs of being in a rationale
                            batch.reviews_tokenized = batch.reviews_tokenized.to(device)
                            att_pred, token_att = bb_model(**batch.reviews_tokenized)

                            hard_mask = rationale_selector(
                                token_att = token_att,
                                input_ids = batch.reviews_tokenized.input_ids
                            )

                            rationale, _, replace_ratio = validation_rationale_extractor(
                                batch = batch,
                                hard_mask = hard_mask
                            )

                            # predict from rationale
                            hard_pred = rp_model(**rationale)
            
                            bb_valid = bb_model.get_loss(
                                att_pred = att_pred,
                                hard_pred = hard_pred,
                                labels = batch.labels_bb.to(device),
                                proximity = proximity
                            )

                            rp_valid = rp_model.get_loss(
                                att_pred = att_pred,
                                hard_pred = hard_pred,
                                labels = batch.labels_rp.to(device),
                                proximity = proximity
                            )

                            bb_running_valid_loss += bb_valid.item()
                            rp_running_valid_loss += rp_valid.item()
                            running_valid_replace_ratio += replace_ratio

                    # evaluation
                    bb_average_train_loss = bb_running_train_loss / eval_every
                    rp_average_train_loss = rp_running_train_loss / eval_every
                    average_train_replace_ratio = running_train_replace_ratio / eval_every

                    bb_average_valid_loss = bb_running_valid_loss / len(valid_loader)
                    rp_average_valid_loss = rp_running_valid_loss / len(valid_loader)
                    average_valid_replace_ratio = running_valid_replace_ratio / len(valid_loader)

                    # bb_improved = bb_best_train_loss > bb_average_train_loss and bb_best_valid_loss > bb_average_valid_loss
                    # rp_improved = rp_best_train_loss > rp_average_train_loss and rp_best_valid_loss > rp_average_valid_loss
                    bb_improved = bb_best_valid_loss > bb_average_valid_loss
                    rp_improved = rp_best_valid_loss > rp_average_valid_loss

                    patience_left = patience if bb_improved and rp_improved else patience_left - 1

                    metrics.append({
                        "bb": {
                            "train_loss": bb_average_train_loss,
                            "valid_loss": bb_average_valid_loss
                        },
                        "rp": {
                            "train_loss": rp_average_train_loss,
                            "valid_loss": rp_average_valid_loss
                        },
                        "replace_ratio": {
                            "replace_train_ratio": average_train_replace_ratio,
                            "replace_valid_ratio": average_valid_replace_ratio
                        },
                        "patience_left": patience_left,
                        "step": global_step,
                    })

                    # update running values
                    bb_best_train_loss = min(bb_best_train_loss, bb_average_train_loss)
                    bb_best_valid_loss = min(bb_best_valid_loss, bb_average_valid_loss)
                    rp_best_train_loss = min(rp_best_train_loss, rp_average_train_loss)
                    rp_best_valid_loss = min(rp_best_valid_loss, rp_average_valid_loss)

                    # resetting running values
                    bb_running_train_loss = 0.0
                    rp_running_train_loss = 0.0
                    running_train_replace_ratio = 0.0
                    bb_running_valid_loss = 0.0
                    rp_running_valid_loss = 0.0
                    running_valid_replace_ratio = 0.0

                    # print progress
                    pb.write(f'Epoch [{epoch+1}/{num_epochs}], Step [{global_step}/{num_epochs*len(train_loader)}]')
                    pb.write(f'Train Probability of Replacement: {average_train_replace_ratio * 100:.4f}')
                    pb.write(f'Valid Probability of Replacement: {average_valid_replace_ratio * 100:.4f}')
                    pb.write(f'BB Train Loss: {bb_average_train_loss:.4f}, BB Valid Loss: {bb_average_valid_loss:.4f}')
                    pb.write(f'RP Train Loss: {rp_average_train_loss:.4f}, RP Valid Loss: {rp_average_valid_loss:.4f}')
                    pb.write(f"Patience: {patience_left}")

                    # checkpoint 
                    if bb_improved and rp_improved:
                        pb.write(f'Model saved to ==> {bb_model_save(bb_model, checkpoint_dir)}')
                        pb.write(f'Model saved to ==> {rp_model_save(rp_model, checkpoint_dir)}')
                    pb.write(f'Metrics saved to ==> {metrics_save(metrics, checkpoint_dir)}')
                    pb.write(f'Epoch copy ==> {bb_model_save_epoch(bb_model, checkpoint_dir, num_epochs, selection_method)}')
                    pb.write(f'Epoch copy ==> {rp_model_save_epoch(rp_model, checkpoint_dir, num_epochs, selection_method)}')
                    pb.write(f'Epoch metrics ==> {metrics_save_epoch(metrics, checkpoint_dir, num_epochs, selection_method)}')

                    bb_model.train()
                    rp_model.train()

            if patience_left == 0:
                break


def evaluate(
    bb_model,
    rp_model,
    tokenizer,
    test_loader,
    device,
    show_detail,
    rationale_selector,
    rationale_extractor,
    checkpoint_dir,
    result_path,
    num_epochs, 
    selection_method
    ):

    if checkpoint_dir is not None:
        print('Loading BB model')
        print('Loading RP model')
        bb_model_load(bb_model, checkpoint_dir, num_epochs, selection_method)
        rp_model_load(rp_model, checkpoint_dir, num_epochs, selection_method)

    if show_detail:
        detail_path = os.path.join(os.path.dirname(checkpoint_dir), "details")
        os.makedirs(detail_path, exist_ok=True)

    gen_spans = 0
    rat_spans = 0
    gen_rat_span_ratio = 0.0
    gen_rat_span_rtotal = 0
    max_gen_span = torch.zeros(1)
    max_rat_span = torch.zeros(1)

    tp = 0
    fp = 0
    fn = 0

    rratio = 0.0

    rprec = 0
    rrec = 0
    rf1 = 0
    rtotal = 0

    y_pred = []
    y_true = []

    comp = []
    suff = []

    ious = []
    num_gen_tokens = []
    num_rat_tokens = []

    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    span_metrics = {
    t: {
        "eraser_tp": 0,
        "one_tp": 0,
        "length_C": 0,
        "at_tp": 0,
        "pred_spans": 0,
        "gold_spans": 0,
        "pred_tokens": 0,
        "gold_tokens": 0
    } for t in iou_thresholds
    }

    macro_metrics = {
    t: {
        "eraser": [],
        "one": [],
        "length": [],
        "atio": []
    } for t in iou_thresholds
}

    review_count = 0

    bb_model.eval()
    rp_model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch.reviews_tokenized = batch.reviews_tokenized.to(device)
            # generate prediction and token probs of being in a rationale
            _, token_att = bb_model(**batch.reviews_tokenized)
            # mask based on probs
            hard_mask = rationale_selector(
                token_att = token_att,
                input_ids = batch.reviews_tokenized.input_ids
            )
            # apply mask and recover rationale
            rationale, remainder, replace_ratio = rationale_extractor(batch, hard_mask)
            rratio += replace_ratio
            # predict from rationale
            hard_pred_logits = rp_model(**rationale)
            hard_pred_probs = torch.sigmoid(hard_pred_logits)

            y_pred.extend(torch.argmax(hard_pred_logits, 1).tolist())
            y_true.extend(batch.labels)

            label_pred_probs = get_label_pred_probs(hard_pred_probs, batch.labels)

            remainder_hard_pred_probs = torch.sigmoid(rp_model(**remainder))
            remainder_label_pred_probs = get_label_pred_probs(remainder_hard_pred_probs, batch.labels)

            all_hard_pred_probs = torch.sigmoid(rp_model(**batch.reviews_tokenized))
            all_label_pred_probs = get_label_pred_probs(all_hard_pred_probs, batch.labels)

            comp.extend((all_label_pred_probs - remainder_label_pred_probs).tolist())
            suff.extend((all_label_pred_probs - label_pred_probs).tolist())

            for i in range(hard_mask.shape[0]):
                gen_mask = torch.tensor([False] + hard_mask[i, :, :].squeeze().tolist())
                rat_mask = torch.tensor([any(id in range(low, high) for (low, high) in batch.rationale_ranges[i]) for id in batch.reviews_tokenized.word_ids(i)])

                gen_span = torch.logical_and(gen_mask[:-1] == False, gen_mask[1:] == True).sum()
                gen_spans += gen_span
                max_gen_span = torch.max(torch.stack([max_gen_span.squeeze(), gen_span.squeeze()]))
                rat_span = torch.logical_and(rat_mask[:-1] == False, rat_mask[1:] == True).sum()
                max_rat_span = torch.max(torch.stack([max_rat_span.squeeze(), rat_span.squeeze()]))
                rat_spans += rat_span
                if not rat_span == torch.zeros(1): # rat_span is LongTensor, so it works
                    gen_rat_span_ratio += gen_span/(rat_span)
                    gen_rat_span_rtotal += 1

                rtp = torch.sum(gen_mask & rat_mask).item()
                tp += rtp
                rfn = torch.sum(~gen_mask & rat_mask).item()
                fn += rfn
                rfp = torch.sum(gen_mask & ~rat_mask).item()
                fp += rfp

                rtotal += 1
                rprec += rtp/(rtp + rfp + 1e-6)
                rrec += rtp/(rtp + rfn + 1e-6)
                rf1 += rtp/(rtp + ((rfp + rfn)/2) + 1e-6)

                if show_detail:
                    with open(os.path.join(detail_path, f"review_{review_count}.txt"), "w") as f:
                        review_tokens = tokenizer.convert_ids_to_tokens(batch.reviews_tokenized.input_ids[i])
                        review_tokens_colored = [color_token(token, g, h) for token, g, h in zip(review_tokens, gen_mask, rat_mask)]
                        print(" ".join(review_tokens_colored), file = f)
                        print(f"Class: {'POSITIVE' if batch.labels[i] else 'NEGATIVE'}", file = f)
                        print(f"P: {100*rtp/(rtp + rfp + 1e-6):.2f} R: {100*rtp/(rtp + rfn + 1e-6):.2f} F1: {100*rtp/(rtp + ((rfp + rfn)/2) + 1e-6):.2f}", file = f)
                    review_count += 1

                gen_sets = to_sets(to_ranges(gen_mask))
                num_gen_tokens.append(sum([len(s) for s in gen_sets]))
                rat_sets = to_sets(to_ranges(rat_mask))
                num_rat_tokens.append(sum([len(s) for s in rat_sets]))


                # ===== spans =====
                # gen_sets = to_sets(to_ranges(gen_mask))
                # rat_sets = to_sets(to_ranges(rat_mask))

                TR = sum(len(s) for s in gen_sets)
                TA = sum(len(s) for s in rat_sets)

                # ===== rious (your notation) =====
                rious = [
                    (
                        max([len(g & r) / (len(g | r) + 1e-6) for r in rat_sets] + [0.0]),
                        len(g)
                    )
                    for g in gen_sets
                ]

                ious.append(rious)

                # ===== SPAN METRICS =====
                for t in iou_thresholds:

                    matched = set()

                    for idx, (riou, rlen) in enumerate(rious):

                        # ERASER
                        # if riou >= t:
                        #     span_metrics[t]["eraser_tp"] += 1
                        # ERASER partial_match

                        for r in rat_sets:

                            iou = len(gen_sets[idx] & r) / (len(gen_sets[idx] | r) + 1e-6)

                            if iou >= t:
                                span_metrics[t]["eraser_tp"] += 1
                                break

                        # LENGTH
                        if riou >= t:
                            span_metrics[t]["length_C"] += rlen

                        # find best match
                        best_idx = -1
                        best_val = 0

                        for j, a in enumerate(rat_sets):
                            val = len(gen_sets[idx] & a) / (len(gen_sets[idx] | a) + 1e-6)
                            if val > best_val:
                                best_val = val
                                best_idx = j

                        # ONE-TO-ONE + AT-IoU
                        if best_idx != -1 and riou >= t and best_idx not in matched:

                            span_metrics[t]["one_tp"] += 1

                            overlap = len(gen_sets[idx] & rat_sets[best_idx])
                            span_metrics[t]["at_tp"] += overlap

                            matched.add(best_idx)

                    # totals
                    span_metrics[t]["pred_spans"] += len(gen_sets)
                    span_metrics[t]["gold_spans"] += len(rat_sets)
                    span_metrics[t]["pred_tokens"] += TR
                    span_metrics[t]["gold_tokens"] += TA

                    # ===== CORRECT MACRO (PER-SAMPLE) =====

                    sample_metrics = {
                        "eraser_tp": 0,
                        "one_tp": 0,
                        "length_C": 0,
                        "at_tp": 0
                    }

                    matched_macro = set()

                    for idx, (riou, rlen) in enumerate(rious):

                        for r in rat_sets:

                            iou = len(gen_sets[idx] & r) / (len(gen_sets[idx] | r) + 1e-6)

                            if iou >= t:
                                sample_metrics["eraser_tp"] += 1
                                break
                        # ERASER + LENGTH
                        if riou >= t:
                            # sample_metrics["eraser_tp"] += 1
                            sample_metrics["length_C"] += rlen

                        # find best match
                        best_idx = -1
                        best_val = 0

                        for j, a in enumerate(rat_sets):
                            val = len(gen_sets[idx] & a) / (len(gen_sets[idx] | a) + 1e-6)
                            if val > best_val:
                                best_val = val
                                best_idx = j

                        # ONE-TO-ONE + AT-IoU (FIXED)
                        if best_idx != -1 and riou >= t and best_idx not in matched_macro:

                            sample_metrics["one_tp"] += 1

                            overlap = len(gen_sets[idx] & rat_sets[best_idx])
                            sample_metrics["at_tp"] += overlap

                            matched_macro.add(best_idx)

                    # compute per-sample precision/recall
                    eraser_p_i = sample_metrics["eraser_tp"] / (len(gen_sets) + 1e-6)
                    eraser_r_i = sample_metrics["eraser_tp"] / (len(rat_sets) + 1e-6)

                    one_p_i = sample_metrics["one_tp"] / (len(gen_sets) + 1e-6)
                    one_r_i = sample_metrics["one_tp"] / (len(rat_sets) + 1e-6)

                    lw_p_i = sample_metrics["length_C"] / (TR + 1e-6)
                    lw_r_i = sample_metrics["length_C"] / (TA + 1e-6)

                    at_p_i = sample_metrics["at_tp"] / (TR + 1e-6)
                    at_r_i = sample_metrics["at_tp"] / (TA + 1e-6)

                    # store
                    macro_metrics[t]["eraser"].append((eraser_p_i, eraser_r_i))
                    macro_metrics[t]["one"].append((one_p_i, one_r_i))
                    macro_metrics[t]["length"].append((lw_p_i, lw_r_i))
                    macro_metrics[t]["atio"].append((at_p_i, at_r_i))

    micro_prec = tp/(tp + fp)
    micro_rec = tp/(tp + fn)
    micro_f1 = tp/(tp + ((fp + fn)/2))
    macro_prec = rprec/rtotal
    macro_rec = rrec/rtotal
    macro_f1 = rf1/rtotal

    micro_iou = dict()
    macro_iou = dict()
    
    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    for threshold in iou_thresholds:

        thresholded_ious = []

        for r in ious:   # ✅ use stored list of all samples
            val = sum([int(riou >= threshold) * riou_tokens for riou, riou_tokens in r])
            thresholded_ious.append(val)

        micro_iou[threshold] = dict()
        micro_iou[threshold]["prec"] = sum(thresholded_ious) / sum(num_gen_tokens)
        micro_iou[threshold]["rec"] = sum(thresholded_ious) / sum(num_rat_tokens)
        micro_iou[threshold]["f1"] = (2 * micro_iou[threshold]["prec"] * micro_iou[threshold]["rec"])/(micro_iou[threshold]["prec"] + micro_iou[threshold]["rec"])

        iou_rprec = [x/(y + 1e-6) for x,y in zip(thresholded_ious, num_gen_tokens)]
        iou_rrec = [x/(y + 1e-6) for x,y in zip(thresholded_ious, num_rat_tokens)]
        macro_iou[threshold] = dict()
        macro_iou[threshold]["prec"] = sum(iou_rprec) / len(iou_rprec)
        macro_iou[threshold]["rec"] = sum(iou_rrec) / len(iou_rrec)
        macro_iou[threshold]["f1"] = (2 * macro_iou[threshold]["prec"] * macro_iou[threshold]["rec"])/(macro_iou[threshold]["prec"] + macro_iou[threshold]["rec"])

    def macro_avg(lst):
        p = sum(x[0] for x in lst) / len(lst)
        r = sum(x[1] for x in lst) / len(lst)
        f1 = (2*p*r)/(p+r+1e-6)
        return p, r, f1



    print("\n===== FINAL SPAN IoU METRICS =====")

    for t in iou_thresholds:

        m = span_metrics[t]

        # MICRO
        eraser_p = m["eraser_tp"] / (m["pred_spans"] + 1e-6)
        eraser_r = m["eraser_tp"] / (m["gold_spans"] + 1e-6)
        eraser_f1 = (2 * eraser_p * eraser_r) / (eraser_p + eraser_r + 1e-6)

        one_p = m["one_tp"] / (m["pred_spans"] + 1e-6)
        one_r = m["one_tp"] / (m["gold_spans"] + 1e-6)
        one_f1 = (2 * one_p * one_r) / (one_p + one_r + 1e-6)

        lw_p = m["length_C"] / (m["pred_tokens"] + 1e-6)
        lw_r = m["length_C"] / (m["gold_tokens"] + 1e-6)
        lw_f1 = (2 * lw_p * lw_r) / (lw_p + lw_r + 1e-6)

        at_p = m["at_tp"] / (m["pred_tokens"] + 1e-6)
        at_r = m["at_tp"] / (m["gold_tokens"] + 1e-6)
        at_f1 = (2 * at_p * at_r) / (at_p + at_r + 1e-6)

        # MACRO
        eraser_mp, eraser_mr, eraser_mf1 = macro_avg(macro_metrics[t]["eraser"])
        one_mp, one_mr, one_mf1 = macro_avg(macro_metrics[t]["one"])
        lw_mp, lw_mr, lw_mf1 = macro_avg(macro_metrics[t]["length"])
        at_mp, at_mr, at_mf1 = macro_avg(macro_metrics[t]["atio"])

        print(f"\nThreshold {t}")

        print(f"ERASER MICRO : P={eraser_p:.4f}, R={eraser_r:.4f}, F1={eraser_f1:.4f}")
        print(f"ERASER MACRO : P={eraser_mp:.4f}, R={eraser_mr:.4f}, F1={eraser_mf1:.4f}")

        print(f"One2One MICRO : P={one_p:.4f}, R={one_r:.4f}, F1={one_f1:.4f}")
        print(f"One2One MACRO : P={one_mp:.4f}, R={one_mr:.4f}, F1={one_mf1:.4f}")

        print(f"Length MICRO : P={lw_p:.4f}, R={lw_r:.4f}, F1={lw_f1:.4f}")
        print(f"Length MACRO : P={lw_mp:.4f}, R={lw_mr:.4f}, F1={lw_mf1:.4f}")

        print(f"AT-IoU MICRO : P={at_p:.4f}, R={at_r:.4f}, F1={at_f1:.4f}")
        print(f"AT-IoU MACRO : P={at_mp:.4f}, R={at_mr:.4f}, F1={at_mf1:.4f}")

    results = {
        "rationales": {
            "micro": {"prec": micro_prec, "rec": micro_rec, "F1": micro_f1},
            "macro": {"prec": macro_prec, "rec": macro_rec, "F1": macro_f1},
        },
        "token_selector_sparsity": rationale_selector.sparsity,
        "replace_ratio": rratio/len(test_loader),
        "comp_suff": {"comprehensiveness": sum(comp)/rtotal, "sufficiency": sum(suff)/rtotal},
        "macro_iou": macro_iou,
        "micro_iou": micro_iou,
        "accuracy": classification_report(y_true, y_pred, labels=[1,0], digits=4, output_dict=True)["accuracy"]
    }

    if not show_detail:
        save_results(results, result_path)  # keep original
        save_results_epoch(results, result_path, num_epochs, selection_method)

    print("Rationales:")


    print(f"Token-level Micro-Averaged Precision: {micro_prec:.4f} Recall: {micro_rec:.4f} F1: {micro_f1:.4f}")
    print(f"Token-level Macro-Averaged Precision: {macro_prec:.4f} Recall: {macro_rec:.4f} F1: {macro_f1:.4f}")
    for t in iou_thresholds:
        print(f"Token-level IOU Micro-Averaged Precision (threshold={t}): {micro_iou[t]['prec']:.4f} Recall: {micro_iou[t]['rec']:.4f} F1: {micro_iou[t]['f1']:.4f}")
        print(f"Token-level IOU Macro-Averaged Precision (threshold={t}): {macro_iou[t]['prec']:.4f} Recall: {macro_iou[t]['rec']:.4f} F1: {macro_iou[t]['f1']:.4f}")
    print(f"Replacement Ratio: {rratio/len(test_loader)}")
    print(f"Average number of generated spans: {gen_spans/rtotal:.4f}, labeled rationale spans: {rat_spans/rtotal:.4f}")
    print(f"Maximum number of generated spans: {max_gen_span:.0f}, labeled rationale spans: {max_rat_span:.0f}")
    print(f"Average ratio of generated spans to labeled rationale spans: {gen_rat_span_ratio/gen_rat_span_rtotal:.4f}")
    print(f"Comprehensiveness: {sum(comp)/rtotal:.4f}")
    print(f"Sufficiency: {sum(suff)/rtotal:.4f}")
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))


def save_results(results, result_path):
    with open(os.path.join(result_path, "results.json"), "w") as f:
        json.dump(results, f)


def to_ranges(mask):
    t1 = torch.tensor(mask.tolist() + [0], dtype=torch.bool)
    t2 = torch.tensor([0] + mask.tolist(), dtype=torch.bool)
    start = torch.logical_and(t1, ~t2)
    end = torch.logical_and(~t1, t2)
    indices = torch.arange(len(t1))
    return list(zip(indices[start].tolist(), indices[end].tolist()))


def to_sets(ranges):
    return [set(range(low, high)) for low, high in ranges]


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def get_label_pred_probs(pred_probs, labels):
    return torch.tensor([pred_prob[label] for pred_prob, label in zip(pred_probs, labels)])


def color_token(token, generated, handlabeled):
    if generated and handlabeled:
        return colored(token, "green")
    if not generated and handlabeled:
        return colored(token, "blue")
    if generated and not handlabeled:
        return colored(token, "red")
    return token


def model_save(model, path):
    torch.save(model.state_dict(), path)


def bb_model_save(model, path):
    save_path = os.path.join(path, "bb_model.pt")
    model_save(model, save_path)
    return save_path


def rp_model_save(model, path):
    save_path = os.path.join(path, "rp_model.pt")
    model_save(model, save_path)
    return save_path


def model_load(model, path):
    return model.load_state_dict(torch.load(path))


def bb_model_load(model, path, num_epochs, selection_method):
    model_load(model, os.path.join(path, "bb_model.pt"))

def rp_model_load(model, path, num_epochs, selection_method):
    model_load(model, os.path.join(path, "rp_model.pt"))


def metrics_save(metrics, path):
    save_path = os.path.join(path, "metrics.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f)
    return save_path

# ===== NEW (DO NOT MODIFY OLD FUNCTIONS) =====
def bb_model_save_epoch(model, path, num_epochs, selection_method):
    save_path = os.path.join(path, f"bb_model_{selection_method}_epoch{num_epochs}.pt")
    model_save(model, save_path)
    return save_path

def rp_model_save_epoch(model, path, num_epochs, selection_method):
    save_path = os.path.join(path, f"rp_model_{selection_method}_epoch{num_epochs}.pt")
    model_save(model, save_path)
    return save_path

def metrics_save_epoch(metrics, path, num_epochs, selection_method):
    save_path = os.path.join(path, f"metrics_{selection_method}_epoch{num_epochs}.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f)
    return save_path

def save_results_epoch(results, result_path, num_epochs, selection_method):
    save_path = os.path.join(result_path, f"results_{selection_method}_epoch{num_epochs}.json")
    with open(save_path, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main(parse_args())
