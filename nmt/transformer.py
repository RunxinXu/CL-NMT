import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import math
import torch.nn.functional as F
from dataset import subsequent_mask

# Refers to: https://nlp.seas.harvard.edu/2018/04/03/attention.html
# But we add cache mechanism in Transformer Decoder => TODO

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    src_embedding = Embeddings(d_model, src_vocab)
    trg_embedding = Embeddings(d_model, tgt_vocab)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(src_embedding, c(position)),
        nn.Sequential(trg_embedding, c(position)),
        Generator(d_model, tgt_vocab, weight=trg_embedding.lut.weight))
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.generator(self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask))
    
    def encode(self, src, src_mask):
        # src: bsz * seq_len 
        # src_mask: bsz * seq_len
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        # memory: bsz * src_seq_len * d_model
        # src_mask: bsz * src_seq_len
        # tgt: bsz * tgt_seq_len
        # tgt_mask: bsz * tgt_seq_len
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def greedy_decode(self, src, src_mask, max_decode_length=256):
        # for inference
        BOS, EOS = 1, 2
        bsz, src_seq_len = src.size()
        results = [[] for _ in range(bsz)] # 放着的是token_ids
        exit_flag = [False for _ in range(bsz)]
        finish_count = 0
        
        memory = self.encoder(self.src_embed(src), src_mask)
        trg = memory.new_full(size=(bsz, 1), fill_value=BOS, dtype=torch.long)
        trg_input = self.tgt_embed(trg)
        cache_dict = {}
        for i in range(len(self.decoder.layers)):
            cache_dict['layer_{}_key'.format(i)] = []
            cache_dict['layer_{}_value'.format(i)] = []
        
        # No Cache
#         for step in range(max_decode_length):
#             tgt_mask = subsequent_mask(trg_input.size(1)).expand(bsz, -1, -1).cuda()
#             output = self.decoder(trg_input, memory, src_mask, tgt_mask)[:, -1, :] # bsz * dim
#             logits = self.generator(output) # bsz * vocab
#             prediction = torch.argmax(logits, dim=-1) # bsz
#             new_trg_input = self.tgt_embed(prediction.unsqueeze(1)) # bsz * 1 * dim
#             trg_input = torch.cat((trg_input, new_trg_input), dim=1)
#             prediction = prediction.cpu().numpy()
#             for i in range(bsz):
#                 if not exit_flag[i]:
#                     if prediction[i] == EOS:
#                         finish_count += 1
#                         exit_flag[i] = True
#                     else:
#                         results[i].append(prediction[i].item())
            
#             if finish_count == bsz:
#                 break
        # Cache
        for step in range(max_decode_length):
            # bsz * dim
            output, cache_dict = self.decoder(trg_input, memory, src_mask, tgt_mask=None, cache_dict=cache_dict)
            output = output[:, 0, :] 
            logits = self.generator(output) # bsz * vocab
            prediction = torch.argmax(logits, dim=-1) # bsz
            trg_input = self.tgt_embed(prediction.unsqueeze(1)) # bsz * 1 * dim
            prediction = prediction.cpu().numpy()
            for i in range(bsz):
                if not exit_flag[i]:
                    if prediction[i] == EOS:
                        finish_count += 1
                        exit_flag[i] = True
                    else:
                        results[i].append(prediction[i].item())
            
            if finish_count == bsz:
                break
                
        return results
                
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab, weight=None):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        if weight is not None:
            self.proj.weight = weight

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask, cache_dict=None):
        if cache_dict is None:
            for i, layer in enumerate(self.layers):
                x = layer(x, memory, src_mask, tgt_mask)
            return self.norm(x)
        else:
            for i, layer in enumerate(self.layers):
                x, cache_key, cache_value = layer(x, memory, src_mask, tgt_mask, cache_dict['layer_{}_key'.format(i)], cache_dict['layer_{}_value'.format(i)])
                cache_dict['layer_{}_key'.format(i)] = cache_key
                cache_dict['layer_{}_value'.format(i)] = cache_value
            return self.norm(x), cache_dict

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        y = sublayer(self.norm(x))
        if isinstance(y, tuple):
            return x + self.dropout(y[0]), *(y[1:])
        else:
            return x + self.dropout(y)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask, cache_key=None, cache_value=None):
        if cache_key is None:
            m = memory
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward)
        else:
            m = memory
            x, cache_key, cache_value = self.sublayer[0](x, lambda x: self.self_attn.cache_forward(x, cache_key, cache_value))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward), cache_key, cache_value
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        # src_mask: bsz * 1 * src_seq_len
        # trg_mask: bsz * trg_seq_len * src_seq_len
        # src_mask中间维度可以是1的原因是每个位置的mask是一样的
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
    def cache_forward(self, x, cache_key, cache_value, mask=None):
        "Implements Figure 2"
        # src_mask: bsz * 1 * src_seq_len
        # trg_mask: bsz * trg_seq_len * src_seq_len
        # src_mask中间维度可以是1的原因是每个位置的mask是一样的
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = x.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.linears[0](x)
        key = self.linears[1](x)
        value = self.linears[2](x)
        if cache_key != []:
            cache_key = torch.cat((cache_key, key), dim=1)
            cache_value = torch.cat((cache_value, value), dim=1)
        else:
            cache_key = key
            cache_value = value
        
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (x, x, x))]
        
        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = cache_key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = cache_value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), cache_key, cache_value
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, weight=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)