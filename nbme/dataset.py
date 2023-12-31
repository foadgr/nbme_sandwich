
import numpy as np
import torch
from torch.utils.data import Dataset



def prepare_input(cfg, text, feature_text):
    print(cfg)
    inputs = cfg.tokenizer(text + feature_text,
                           add_special_tokens=True,
                           max_length=cfg.max_len,
                           padding="max_length",
                           truncation=True,
                           return_tensors="pt")
    return inputs


def create_label(cfg, text, annotation_length, location_list):
    encoded = cfg.tokenizer(text,
                            add_special_tokens=True,
                            max_length=cfg.max_len,
                            padding="max_length",
                            return_offsets_mapping=True)
    offset_mapping = encoded['offset_mapping']
    ignore_idxes = np.where(np.array(encoded.sequence_ids()) != 0)[0]
    label = np.zeros(len(offset_mapping))
    label[ignore_idxes] = -1
    if annotation_length != 0:
        for location in location_list:
            for loc in [s.split() for s in location.split(';')]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx
                if (start_idx != -1) & (end_idx != -1):
                    label[start_idx:end_idx] = 1
    return torch.tensor(label, dtype=torch.float)

class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        print(cfg)
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values
        self.annotation_lengths = df['annotation_length'].values
        self.locations = df['location'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, 
                               self.pn_historys[item], 
                               self.feature_texts[item])
        label = create_label(self.cfg, 
                             self.pn_historys[item], 
                             self.annotation_lengths[item], 
                             self.locations[item])
        inputs['labels'] = label # Adding label to inputs
        return inputs