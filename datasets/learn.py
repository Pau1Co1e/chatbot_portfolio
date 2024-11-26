"""To iterate over full datasets use a dataset directly; bypassing allocating or batching the dataset."""
from datasets import load_dataset
from safetensors import torch as safe_torch
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import pandas as pd
import torch


dataset = load_dataset("parquet", data_files={'train': 'squad_v2_train.parquet', 'test': 'squad_v2_test.parquet'})


print(dataset)
print()
print(dataset[0])
dfs = {split: dset.to_pandas() for split, dset in subjqa.flatten().items()}

for split, df in dfs.items():
    print(f"Number of questions in {split}: {df['id'].nunique()}")