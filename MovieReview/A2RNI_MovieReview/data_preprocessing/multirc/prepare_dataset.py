import os
import json
from argparse import ArgumentParser
from nltk.tokenize import word_tokenize


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="multirc")
    parser.add_argument("--save_path", type=str, default="original")
    parser.add_argument("--lower_case", action="store_true")
    return parser.parse_args()


SPLITS = ("train", "val", "test")
SPLITS_OUT = ("train", "valid", "test")


def load_jsonl(f):
    return [json.loads(ex) for ex in f.read().splitlines()]


def get_split_path(path, split):
    return os.path.join(path, f"{split}.jsonl")


def load_jsonl_split(path, split):
    with open(get_split_path(path, split), "r") as f:
        return load_jsonl(f)


def load_examples(path):
    return {split: load_jsonl_split(path, split) for split in SPLITS}


def check_evidences_outer_list_len(example):
    if len(example["evidences"]) > 1:
        print(f"evidences outer list longer than one: {example['docid']}")


def check_evidences_sent_len(example):
    for evidence in example["evidences"][0]:
        if (evidence["end_sentence"] - evidence["start_sentence"]) > 1:
            print(f"evidence longer than one sentence: {example['docid']}")


def check_evidences_format(examples):
    for split in SPLITS:
        for example in examples[split]:
            check_evidences_outer_list_len(example)
            check_evidences_sent_len(example)


def load_doc(path):
    with open(path, "r") as f:
        return f.read().splitlines()


def docs_path(path):
    return os.path.join(path, "docs")


def load_docs(path):
    return {doc.name:load_doc(doc.path) for doc in os.scandir(docs_path(path))}


def tokenize_doc(doc):
    return [word_tokenize(sentence) for sentence in doc]


def tokenize_docs(docs):
    return {name:tokenize_doc(doc) for name, doc in docs.items()}


def to_lower_doc(doc):
    return [[token.lower() for token in sentence] for sentence in doc]


def to_lower_docs(docs):
    return {name:to_lower_doc(doc) for name, doc in docs.items()}


def parse_label(example):
    return 1 if example["classification"] == "True" else 0


def get_rationale_indices(example):
    return [ev["start_sentence"] for ev in example["evidences"][0]]


def get_query(example):
    return [token.lower() for token in example["query"].split()]


def get_example_name_parts(example):
    return example["annotation_id"].split(":")


def get_sentences(example, docs):
    name, _, _ = get_example_name_parts(example)
    return docs[name]


def prepare_example(example, docs):
    return (get_sentences(example, docs), parse_label(example), get_rationale_indices(example), get_query(example))


def prepare_examples_split(examples, docs):
    return [prepare_example(example, docs) for example in examples]


def prepare_examples(examples, docs):
    return {split: prepare_examples_split(examples[split], docs) for split in SPLITS}


def example_to_jsonl(example):
    return json.dumps(example) + "\n"


def save_split(examples, path):
    with open(path, "w") as f:
        for example in examples:
            f.write(example_to_jsonl(example))


def save_examples(examples, path):
    os.makedirs(path, exist_ok=True)
    for split, split_out in zip(SPLITS, SPLITS_OUT):
        split_out_path = get_split_path(path, split_out)
        print(f"Saving {split_out} to ==> {split_out_path}")
        save_split(examples[split], split_out_path)


def main(args):
    print("Loading MultiRC examples...")
    examples = load_examples(args.data_path)

    print("Checking evidences format...")
    check_evidences_format(examples)

    print("Loading docs...")
    docs = load_docs(args.data_path)

    print("Tokenizing docs...")
    tokenized_docs = tokenize_docs(docs)

    if args.lower_case:
        print("Lower-casing docs...")
        tokenized_docs = to_lower_docs(tokenized_docs)

    print("Preparing examples...")
    prepared_examples = prepare_examples(examples, tokenized_docs)

    print(f"Saving examples to {args.save_path}")
    save_examples(prepared_examples, args.save_path)


if __name__ == "__main__":
    main(parse_args())