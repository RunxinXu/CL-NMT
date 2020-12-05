from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sentencepiece as spm
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class NMTDataset(Dataset):
    def __init__(self, src_path, trg_path, src_spm_path, trg_spm_path, threshold=5): 
        self.src_sp = spm.SentencePieceProcessor()
        self.src_sp.load(src_spm_path)
        self.trg_sp = spm.SentencePieceProcessor()
        self.trg_sp.load(trg_spm_path)
        
        # vocabulary
        self.src_vocabs_size = self.src_sp.get_piece_size()
        self.trg_vocabs_size = self.trg_sp.get_piece_size()
        
        # load data
        self.data = self.load_data(src_path, trg_path)
        
        # special token
        self.pad_id = self.src_sp.pad_id() # 0
        self.bos_id = self.src_sp.bos_id() # 1
        self.eos_id = self.src_sp.eos_id() # 2
        self.unk_id = self.src_sp.unk_id() # 3
    
    def load_data(self, src_path, trg_path):
        data = []
        with open(src_path) as f, \
            open(trg_path) as g:
                for (src, trg) in tqdm(zip(f, g)):
                    src = self.src_sp.encode_as_ids(src)
                    trg = self.trg_sp.encode_as_ids(trg)
                    
                    data.append({
                        'src_ids': src,
                        'trg_ids': trg
                    })
        return data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    PAD = 0
    BOS = 1
    EOS = 2
    
    src = [torch.LongTensor(sample['src_ids']) for sample in batch]
    
    trg = [torch.LongTensor([BOS] + sample['trg_ids']) for sample in batch]
    trg_y = [torch.LongTensor(sample['trg_ids'] + [EOS]) for sample in batch]

    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=PAD)
    trg = nn.utils.rnn.pad_sequence(trg, batch_first=True, padding_value=PAD)
    trg_y = nn.utils.rnn.pad_sequence(trg_y, batch_first=True, padding_value=PAD)
    
    src_mask = (src != PAD).bool().unsqueeze(1)
    trg_mask = (trg != PAD).bool().unsqueeze(-2)
    trg_mask = trg_mask & subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    
    data = {
        'src': src,
        'trg': trg,
        'trg_y': trg_y,
        'src_mask': src_mask,
        'trg_mask': trg_mask
    }

    return data

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

if __name__ == '__main__':
    dataset = NMTDataset('../data/train.en', '../data/train.zh', '../spm_model/en.model', '../spm_model/zh.model')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    print('.')
    for batch in dataloader:
        print(batch)
        input()