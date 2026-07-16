import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


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

        self.input_file = os.path.join(data_dir, file_map[mode])

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

        inputs = self.inputs[item]
        masks = self.masks[item]
        labels = self.labels[item]

        return inputs, masks, labels

    def _create_examples(self, balance=False):

        examples = []

        with open(self.input_file, "rt", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if len(line) == 0:
                    continue

                item = json.loads(line)

                text_tokens = item[0]
                label = int(item[1])

                examples.append({
                    "text": text_tokens,
                    "label": label
                })

        print("Dataset: Movie Review")
        print("{} samples has {}".format(self.mode, len(examples)))

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

        if balance:

            random.seed(12252018)

            print("Make the Training dataset class balanced.")

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

            print(
                "After balance training data: {} positive examples, {} negative examples.".format(
                    len(pos_examples),
                    len(neg_examples)
                )
            )

        return examples

    def _convert_single_text(
            self,
            text_tokens,
            max_length,
            word2idx
    ):

        input_ids = []

        if len(text_tokens) > max_length:
            text_tokens = text_tokens[:max_length]

        for word in text_tokens:

            word = word.strip()

            try:
                input_ids.append(word2idx[word])

            except:
                input_ids.append(0)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_length:
            input_ids.append(0)
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

        inputs = self.inputs[item]
        masks = self.masks[item]
        labels = self.labels[item]
        rationales = self.rationales[item]

        return inputs, masks, labels, rationales

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

        print("Dataset: Movie Review")

        with open(annotation_path, "rt", encoding="utf-8") as f:

            for line_id, line in enumerate(f):

                line = line.strip()

                if len(line) == 0:
                    continue

                item = json.loads(line)

                ####################################################
                # Dataset format:
                # [
                #   tokens,
                #   label,
                #   rationale_spans
                # ]
                ####################################################

                text_tokens = item[0]
                label = int(item[1])
                rationale_spans = item[2]

                original_tokens = text_tokens.copy()

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
                        input_ids.append(word2idx[word])

                    except:
                        input_ids.append(0)

                ####################################################
                # mask
                ####################################################

                input_mask = [1] * len(input_ids)

                while len(input_ids) < max_length:
                    input_ids.append(0)
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
                    # rationale outside truncation
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
                # DEBUG FIRST 3 EXAMPLES
                ####################################################

                # if line_id < 3:

                #     print("\n================================================")
                #     print("EXAMPLE:", line_id)
                #     print("================================================")

                #     print("\nLABEL:")
                #     print(label)

                #     print("\nTOTAL ORIGINAL TOKENS:")
                #     print(len(original_tokens))

                #     print("\nTOKENS AFTER TRUNCATION:")
                #     print(len(text_tokens))

                #     print("\nRATIONALE SPANS:")
                #     print(rationale_spans)

                #     selected_tokens = []

                #     for span in rationale_spans:

                #         start = span[0]
                #         end = span[1]

                #         if start >= max_length:
                #             continue

                #         if end > max_length:
                #             end = max_length

                #         selected_tokens.extend(
                #             original_tokens[start:end]
                #         )

                #     print("\nRATIONALE TOKENS:")
                #     print(selected_tokens)

                #     rationale_positions = [
                #         i for i, v in enumerate(binary_rationale)
                #         if v == 1
                #     ]

                #     print("\nRATIONALE POSITIONS:")
                #     print(rationale_positions)

                #     print("\nGOLD RATIONALE DENSITY:")
                #     print(
                #         gold_rationale_tokens / review_length
                #     )

                #     print("\nFIRST 120 TOKENS:")
                #     print(original_tokens[:120])

                #     print("================================================\n")



                ####################################################
                # append
                ####################################################

                data.append(input_ids)
                masks.append(input_mask)
                labels.append(label)
                rationales.append(binary_rationale)

        ############################################################
        # torch tensors
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
        # dataset stats
        ############################################################

        total = self.labels.shape[0]

        print("\nannotation samples has {}".format(total))

        pos = torch.sum(self.labels)
        neg = total - pos

        print(
            "annotation data: {} positive examples, {} negative examples.".format(
                pos,
                neg
            )
        )

        ############################################################
        # GOLD RATIONALE DENSITY
        ############################################################

        gold_density = (
            total_rationale_tokens / total_review_tokens
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

        print("================================================\n")