import os
import argparse
import sentencepiece as spm

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='train', type=str,
                        help="Train or Encode")
    parser.add_argument("--input", default='data/train.en', type=str,
                        help="Input text")
    parser.add_argument("--model_prefix", default='en', type=str,
                        help="Model prefix", )
    parser.add_argument("--vocab_size", default=10000, type=int,
                        help="Vocabulary size" )
    return parser

if __name__ == '__main__':
    args = get_argparse().parse_args()
    spm.SentencePieceTrainer.train(input=args.input, model_prefix=args.model_prefix, vocab_size=args.vocab_size, pad_id=0, unk_id=3)
