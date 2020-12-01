from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sentencepiece as spm
import torch
import torch.nn as nn

class NMTDataset(Dataset):
    def __init__(self, src_path, trg_path, src_spm_path, trg_spm_path, threshold=5): 
        self.src_sp = spm.SentencePieceProcessor()
        self.src_sp.load(src_spm_path)
        self.trg_sp = spm.SentencePieceProcessor()
        self.trg_sp.load(trg_spm_path)
        
        # vocabulary
        src_vocabs = [self.src_sp.id_to_piece(id) for id in range(self.src_sp.get_piece_size())]
        trg_vocabs = [self.trg_sp.id_to_piece(id) for id in range(self.trg_sp.get_piece_size())]
        self.src_vocabs = self.vocab_filter(src_path, self.src_sp, src_vocabs, threshold)
        self.trg_vocabs = self.vocab_filter(trg_path, self.trg_sp, trg_vocabs, threshold)
        self.src_sp.set_vocabulary(self.src_vocabs)
        self.trg_sp.set_vocabulary(self.trg_vocabs)
        self.src_vocabs_size = len(self.src_vocabs)
        self.trg_vocabs_size = len(self.trg_vocabs)
        
        # load data
        self.data = self.load_data(src_path, trg_path)
        
        # special token
        self.pad_id = self.src_sp.pad_id() # 0
        self.bos_id = self.src_sp.bos_id() # 1
        self.eos_id = self.src_sp.eos_id() # 2
        self.unk_id = self.src_sp.unk_id() # 3
        
    def vocab_filter(self, text_path, sp, vocabs, threshold=0):
        freq = {}
        with open(text_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                for piece in sp.encode_as_pieces(line):
                    freq.setdefault(piece, 0)
                    freq[piece] += 1
        new_vocabs = list(filter(lambda x : x in freq and freq[x] > threshold, vocabs))
        return new_vocabs
    
    def load_data(self, src_path, trg_path):
        data = []
        with open(src_path) as f, \
            open(trg_path) as g:
                for (src, trg) in zip(f, g):
                    src, trg = src.strip(), trg.strip()
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

    src = [torch.LongTensor(sample['src_ids']) for sample in batch]
    trg = [torch.LongTensor(sample['trg_ids']) for sample in batch]

    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    trg = nn.utils.rnn.pad_sequence(trg, batch_first=True, padding_value=0)

    data = {
        'src': src,
        'trg': trg,
    }

    return data

if __name__ == '__main__':
    dataset = NMTDataset('../data/train.en', '../data/train.zh', '../spm_model/en.model', '../spm_model/zh.model')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    print('.')
    for batch in dataloader:
        print(batch)
        input()