import torch
import torch.nn as nn
from dataset import NMTDataset, collate_fn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformer import make_model
from tqdm import tqdm
import argparse
from tensorboardX import SummaryWriter

def get_argparse():
    parser = argparse.ArgumentParser()

    # file
    parser.add_argument("--src", default='../data/train.en', type=str,
                        help="src")
    parser.add_argument("--trg", default='../data/train.zh', type=str,
                        help="trg")
    parser.add_argument("--src_spm", default='../spm_model/en.model', type=str,
                        help="src_spm")
    parser.add_argument("--trg_spm", default='../spm_model/zh.model', type=str,
                        help="trg_spm")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="Output folder")
    
    # training
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size" )
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="Batch size" )
    parser.add_argument("--epochs", default=50, type=int,
                        help="Output folder")
    
    # model
    parser.add_argument("--layers", default=6, type=int,
                        help="encoder & decoder layers")
    parser.add_argument("--d_model", default=512, type=int,
                        help="d_model")
    parser.add_argument("--d_ff", default=2048, type=int,
                        help="d_ff")
    parser.add_argument("--heads", default=8, type=int,
                        help="heads")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout")
    
    return parser

def train(train_dataloader, model, args, writer):
    model.train()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    global_step = 0
    for epoch in range(args.epochs):
        for batch in tqdm(train_dataloader):
            src = batch['src'].cuda() # bsz * src_seq_len
            trg = batch['trg'].cuda() # bsz * trg_seq_len
            src_mask = batch['src_mask'].cuda()
            trg_mask = batch['trg_mask'].cuda()
            bsz, trg_len = trg.size()
            output = model(src, trg, src_mask, trg_mask) # bsz * trg_seq_len * vocab
            label = batch['trg_y'].cuda() # bsz * trg_seq_len
            loss = criterion(output.view(bsz*trg_len, -1), label.view(-1))
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1


if __name__ == '__main__':
    args = get_argparse().parse_args()
    dataset = NMTDataset(args.src, args.trg, args.src_spm, args.trg_spm)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    model = make_model(src_vocab=dataset.src_vocabs_size, tgt_vocab=dataset.trg_vocabs_size, N=args.layers, 
               d_model=args.d_model, d_ff=args.d_ff, h=args.heads, dropout=args.dropout)
    model = model.cuda()
    print('total #parameters: {}'.format(sum(p.numel() for p in model.parameters())))
    
    writer = SummaryWriter(args.output_dir)
    train(dataloader, model, args, writer)
