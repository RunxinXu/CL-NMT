import torch
import torch.nn as nn
from dataset import NMTDataset, collate_fn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformer import make_model
from tqdm import tqdm
import argparse
from tensorboardX import SummaryWriter
import sentencepiece as spm
import os
import sacrebleu
import torch.nn.functional as F

# TODO: 
# joint vocabulary?
# beam search?

def get_argparse():
    parser = argparse.ArgumentParser()

    # file
    parser.add_argument("--data", default='../data', type=str,
                        help="src")
    parser.add_argument("--src_spm", default='../spm_model/en.model', type=str,
                        help="src_spm")
    parser.add_argument("--trg_spm", default='../spm_model/zh.model', type=str,
                        help="trg_spm")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="Output folder")
    
    # training
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size")
    parser.add_argument("--test_batch_size", default=128, type=int,
                        help="Batch size")
#     parser.add_argument("--learning_rate", default=1e-3, type=float,
#                         help="Batch size" )
    parser.add_argument("--epochs", default=150, type=int,
                        help="Output folder")
    parser.add_argument("--early_stop", default=10, type=int,
                        help="Batch size" )
    
    # model
    parser.add_argument("--layers", default=4, type=int,
                        help="encoder & decoder layers")
    parser.add_argument("--d_model", default=256, type=int,
                        help="d_model")
    parser.add_argument("--d_ff", default=1024, type=int,
                        help="d_ff")
    parser.add_argument("--heads", default=4, type=int,
                        help="heads")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout")
    
    # criterion
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="dropout")
    return parser

def train(train_dataloader, dev_dataloader, model, args, writer, trg_sp):
    model.train()
    my_optimizer = get_std_opt(model)
    criterion = LabelSmoothingLoss(label_smoothing=args.label_smoothing, ignore_index=0, tgt_vocab_size=model.generator.proj.weight.size(0)).cuda()
    global_step = 0
    best_bleu = -1
    eary_stop = args.early_stop
    for epoch in range(args.epochs):
        for batch in tqdm(train_dataloader):
            src = batch['src'].cuda() # bsz * src_seq_len
            trg = batch['trg'].cuda() # bsz * trg_seq_len
            src_mask = batch['src_mask'].cuda() # bsz * 1 * src_seq_len
            trg_mask = batch['trg_mask'].cuda() # bsz * trg_seq_len * trg_seq_len
            bsz, trg_len = trg.size()
            output = model(src, trg, src_mask, trg_mask) # bsz * trg_seq_len * vocab
            label = batch['trg_y'].cuda() # bsz * trg_seq_len
            loss = criterion(output.view(bsz*trg_len, -1), label.view(-1))
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            my_optimizer.optimizer.zero_grad()
            loss.backward()
            my_optimizer.step()
            global_step += 1
        
        
        model.eval()
        r = test(dev_dataloader, model, args, trg_sp)
        model.train()
        
        bleu = r['bleu']
        if bleu > best_bleu:
            best_bleu = bleu
            early_stop = args.early_stop
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best.pth'))
        else:
            early_stop -= 1
        writer.add_scalar('Dev/BLEU', bleu, epoch)

        if early_stop == 0:
            break

def test(test_dataloader, model, args, trg_sp):
    result = {}
    refs = []
    sys = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            src = batch['src'].cuda() # bsz * src_seq_len
            src_mask = batch['src_mask'].cuda() # bsz * 1 * src_seq_len
            raw_trg = batch['raw_trg']
            decode_results = model.greedy_decode(src, src_mask)
            decode_results = [trg_sp.decode_ids(x) for x in decode_results]
            refs.extend(raw_trg)
            sys.extend(decode_results)
    refs = [refs]
    bleu = sacrebleu.corpus_bleu(sys, refs)
    print(bleu.score)
    
    result['bleu'] = bleu.score
    return result


# OPTIMIZER
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# LABEL SMOOTHING
class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')
            
if __name__ == '__main__':
    args = get_argparse().parse_args()
    
    src_sp = spm.SentencePieceProcessor()
    src_sp.load(args.src_spm)
    trg_sp = spm.SentencePieceProcessor()
    trg_sp.load(args.trg_spm)
    
    train_dataset = NMTDataset(os.path.join(args.data, 'train.en'), os.path.join(args.data, 'train.zh'), src_sp, trg_sp)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataset = NMTDataset(os.path.join(args.data, 'dev.en'), os.path.join(args.data, 'dev.zh'), src_sp, trg_sp)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataset = NMTDataset(os.path.join(args.data, 'test.en'), os.path.join(args.data, 'test.zh'), src_sp, trg_sp)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = make_model(src_vocab=dev_dataset.src_vocabs_size, tgt_vocab=dev_dataset.trg_vocabs_size, N=args.layers, 
               d_model=args.d_model, d_ff=args.d_ff, h=args.heads, dropout=args.dropout)
    model = model.cuda()
    print('total #parameters: {}'.format(sum(p.numel() for p in model.parameters())))
    writer = SummaryWriter(args.output_dir)
    
    train(train_dataloader, test_dataloader, model, args, writer, trg_sp)
