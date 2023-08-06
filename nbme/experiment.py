import os
import ast
import warnings
warnings.filterwarnings("ignore")

from tqdm.auto import tqdm

import torch
import transformers
from transformers import AutoTokenizer
from transformers import DistilBertTokenizer, DistilBertForTokenClassification
# %env TOKENIZERS_PARALLELISM=true

device = torch.device('cpu')
OUTPUT_DIR = './output/'

from nbme.logging import get_logger
from nbme.seed import seed_everything

LOGGER = get_logger(filename='./output/train')
seed_everything(seed=42)

import pandas as pd
from nbme.fix import fix_annotation
from ast import literal_eval

patient_notes = pd.read_csv('./data/patient_notes.csv')
train = pd.read_csv('./data/train.csv')
train['annotation'] = train['annotation'].apply(literal_eval)
train['location'] = train['location'].apply(literal_eval)
features = pd.read_csv('./data/features.csv')

train = pd.merge(train, features, on=['case_num', 'feature_num'], how='left')
train = pd.merge(train, patient_notes, on=['pn_num', 'case_num'], how='left')
train = fix_annotation(train)
train['annotation_length'] = train['annotation'].apply(len)

from sklearn.model_selection import GroupKFold
Fold = GroupKFold(n_splits=CFG.n_fold)
groups = train['pn_num'].values
for n, (train_index, val_index) in enumerate(Fold.split(train, train['location'], groups)):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)

tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
CFG.tokenizer = tokenizer

for text_col in ['pn_history']:
    pn_history_lengths = []
    tk0 = tqdm(patient_notes[text_col].fillna("").values, total=len(patient_notes))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        pn_history_lengths.append(length)
    LOGGER.info(f'{text_col} max(lengths): {max(pn_history_lengths)}')

for text_col in ['feature_text']:
    features_lengths = []
    tk0 = tqdm(features[text_col].fillna("").values, total=len(features))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        features_lengths.append(length)
    LOGGER.info(f'{text_col} max(lengths): {max(features_lengths)}')

CFG.max_len = max(pn_history_lengths) + max(features_lengths) + 3
LOGGER.info(f"max_len: {CFG.max_len}")

from nbme.trainingloop import TrainingLoop
from nbme.scoring import get_char_probs, get_results, get_predictions, create_labels_for_scoring

def get_result(oof_df, CFG):
        labels = create_labels_for_scoring(oof_df)
        predictions = oof_df[[i for i in range(CFG.max_len)]].values
        char_probs = get_char_probs(oof_df['pn_history'].values, predictions, CFG.tokenizer)
        results = get_results(char_probs, th=0.5)
        preds = get_predictions(results)
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}')

training_loop = TrainingLoop(folds=train, config=CFG, device=device) # Adjust the parameters as needed
for fold in range(CFG.n_fold):
    if fold in CFG.trn_fold:
        _oof_df = training_loop.train_fold(fold)
        oof_df = pd.concat([oof_df, _oof_df])
        LOGGER.info(f"========== fold: {fold} result ==========")
        get_result(_oof_df, CFG)