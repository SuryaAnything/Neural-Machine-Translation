import torch
import spacy
from torchtext.data import Field, Example, Dataset, TabularDataset, BucketIterator


# Function to create examples
def create_dataset(de_path, en_path, fields):
    examples = []
    with open(de_path, 'r', encoding='utf-8') as de_file, open(en_path, 'r', encoding='utf-8') as en_file:
        for de_line, en_line in zip(de_file, en_file):
            src = de_line.strip()
            trg = en_line.strip()
            examples.append(Example.fromlist([src, trg], fields))
    return examples


