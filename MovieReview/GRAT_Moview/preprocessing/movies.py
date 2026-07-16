import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


PAD_ID = 1
UNK_ID = 0


class MovieData(Dataset):

    def __init__(
            self,
            data_dir,
            mode,
            word2idx,
            balance=False,
            max_length=256
    ):

        super().__init__()

        self.mode = mode

        file_map = {
            "train": "train.jsonl",
            "dev": "valid.jsonl"
        }

        self.input_file = os.path.join(
            data_dir,
            file_map[mode]
        )

        self.inputs = []
        self.masks = []
        self.labels = []

        examples = self._create_examples(balance)

        self._convert_examples_to_arrays(
            examples,
            max_length,
            word2idx
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):

        return (
            self.inputs[item],
            self.masks[item],
            self.labels[item]
        )

    def _create_examples(self, balance=False):

        examples = []

        with open(self.input_file, "rt", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if len(line) == 0:
                    continue

                item = json.loads(line)

                ####################################################
                # format:
                # [
                #   tokens,
                #   label
                # ]
                ####################################################

                text_tokens = item[0]
                label = int(item[1])

                examples.append({
                    "text": text_tokens,
                    "label": label
                })

        print("\n================================================")
        print("Dataset: Movie Review")
        print("================================================")

        print(
            "{} samples has {}".format(
                self.mode,
                len(examples)
            )
        )

        pos_examples = [
            example for example in examples
            if example["label"] == 1
        ]

        neg_examples = [
            example for example in examples
            if example["label"] == 0
        ]

        print(
            "{} data: {} positive examples, {} negative examples.".format(
                self.mode,
                len(pos_examples),
                len(neg_examples)
            )
        )

        ############################################################
        # balance training dataset
        ############################################################

        if balance:

            random.seed(12252018)

            print("\nMake the Training dataset class balanced.")

            min_examples = min(
                len(pos_examples),
                len(neg_examples)
            )

            if len(pos_examples) > min_examples:

                pos_examples = random.sample(
                    pos_examples,
                    min_examples
                )

            if len(neg_examples) > min_examples:

                neg_examples = random.sample(
                    neg_examples,
                    min_examples
                )

            examples = pos_examples + neg_examples

            random.shuffle(examples)

            print(
                "After balance: {} positive examples, {} negative examples.".format(
                    len(pos_examples),
                    len(neg_examples)
                )
            )

        print("================================================\n")

        return examples

    def _convert_single_text(
            self,
            text_tokens,
            max_length,
            word2idx
    ):

        ############################################################
        # truncate
        ############################################################

        if len(text_tokens) > max_length:

            text_tokens = text_tokens[:max_length]

        ############################################################
        # token -> ids
        ############################################################

        input_ids = []

        for word in text_tokens:

            word = word.strip()

            try:
                input_ids.append(
                    word2idx[word]
                )

            except:
                input_ids.append(
                    UNK_ID
                )

        ############################################################
        # mask
        ############################################################

        input_mask = [1] * len(input_ids)

        ############################################################
        # padding
        ############################################################

        while len(input_ids) < max_length:

            input_ids.append(PAD_ID)
            input_mask.append(0)

        assert len(input_ids) == max_length
        assert len(input_mask) == max_length

        return input_ids, input_mask

    def _convert_examples_to_arrays(
            self,
            examples,
            max_length,
            word2idx
    ):

        data = []
        masks = []
        labels = []

        for example in examples:

            input_ids, input_mask = self._convert_single_text(
                example["text"],
                max_length,
                word2idx
            )

            data.append(input_ids)
            masks.append(input_mask)
            labels.append(example["label"])

        self.inputs = torch.from_numpy(
            np.array(data)
        )

        self.masks = torch.from_numpy(
            np.array(masks)
        )

        self.labels = torch.from_numpy(
            np.array(labels)
        )


class MovieAnnotation(Dataset):

    def __init__(
            self,
            annotation_path,
            word2idx,
            max_length=256
    ):

        super().__init__()

        self.inputs = []
        self.masks = []
        self.labels = []
        self.rationales = []

        self._create_examples(
            annotation_path,
            word2idx,
            max_length
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, item):

        return (
            self.inputs[item],
            self.masks[item],
            self.labels[item],
            self.rationales[item]
        )

    def _create_examples(
            self,
            annotation_path,
            word2idx,
            max_length
    ):

        data = []
        masks = []
        labels = []
        rationales = []

        total_rationale_tokens = 0
        total_review_tokens = 0

        print("\n================================================")
        print("Dataset: Movie Annotation")
        print("================================================")

        with open(annotation_path, "rt", encoding="utf-8") as f:

            for line_id, line in enumerate(f):

                line = line.strip()

                if len(line) == 0:
                    continue

                item = json.loads(line)

                ####################################################
                # format:
                # [
                #   tokens,
                #   label,
                #   rationale_spans
                # ]
                ####################################################

                text_tokens = item[0]
                label = int(item[1])
                rationale_spans = item[2]

                ####################################################
                # truncate review
                ####################################################

                if len(text_tokens) > max_length:

                    text_tokens = text_tokens[:max_length]

                ####################################################
                # token -> ids
                ####################################################

                input_ids = []

                for word in text_tokens:

                    word = word.strip()

                    try:
                        input_ids.append(
                            word2idx[word]
                        )

                    except:
                        input_ids.append(
                            UNK_ID
                        )

                ####################################################
                # mask
                ####################################################

                input_mask = [1] * len(input_ids)

                ####################################################
                # padding
                ####################################################

                while len(input_ids) < max_length:

                    input_ids.append(PAD_ID)
                    input_mask.append(0)

                review_length = sum(input_mask)

                assert len(input_ids) == max_length
                assert len(input_mask) == max_length

                ####################################################
                # rationale mask
                ####################################################

                binary_rationale = [0] * max_length

                for span in rationale_spans:

                    start = span[0]
                    end = span[1]

                    ################################################
                    # rationale completely outside truncation
                    ################################################

                    if start >= max_length:
                        continue

                    ################################################
                    # clip rationale
                    ################################################

                    if end > max_length:
                        end = max_length

                    ################################################
                    # [start, end)
                    ################################################

                    for idx in range(start, end):

                        binary_rationale[idx] = 1

                ####################################################
                # statistics
                ####################################################

                gold_rationale_tokens = sum(binary_rationale)

                total_rationale_tokens += gold_rationale_tokens
                total_review_tokens += review_length

                ####################################################
                # append
                ####################################################

                data.append(input_ids)
                masks.append(input_mask)
                labels.append(label)
                rationales.append(binary_rationale)

        ############################################################
        # tensors
        ############################################################

        self.inputs = torch.from_numpy(
            np.array(data)
        )

        self.masks = torch.from_numpy(
            np.array(masks)
        )

        self.labels = torch.from_numpy(
            np.array(labels)
        )

        self.rationales = torch.from_numpy(
            np.array(rationales)
        )

        ############################################################
        # statistics
        ############################################################

        total = self.labels.shape[0]

        pos = torch.sum(self.labels)
        neg = total - pos

        gold_density = (
            total_rationale_tokens /
            total_review_tokens
        )

        print(
            "annotation samples has {}".format(total)
        )

        print(
            "annotation data: {} positive examples, {} negative examples.".format(
                pos,
                neg
            )
        )

        print("\n================================================")
        print("GOLD RATIONALE STATISTICS")
        print("================================================")

        print(
            "Total rationale tokens:",
            total_rationale_tokens
        )

        print(
            "Total valid review tokens:",
            total_review_tokens
        )

        print(
            "Gold rationale density: {:.6f}".format(
                gold_density
            )
        )

        print(
            "Gold rationale percentage: {:.2f}%".format(
                gold_density * 100
            )
        )

        print(
            "Average rationale tokens per review: {:.4f}".format(
                total_rationale_tokens / total
            )
        )

        print("================================================\n")