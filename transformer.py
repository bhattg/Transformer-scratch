import torch
import torch.nn as nn
import torch.nn.functional as F
import modules
import math
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from decay_rnn import DECAY_RNN
from RIM import RIM 
from ordered_neuron  import ONLSTM

__author__ = "Gantavya Bhatt, taken from Ke Tran (Thanks :D )!!"


def add_timing_signal(x, max_timescale=1e4):
    batch, length, channels = x.size()
    nts = channels // 2
    log_inc = math.log(max_timescale) / nts
    log_inv_inc = -log_inc * torch.arange(0, nts).float()
    inv_inc = log_inv_inc.exp().view(1, -1).expand(length, nts).float()
    pos_idx = torch.arange(0, length).view(-1, 1).expand(length, channels).float()
    pos_emb = torch.FloatTensor(length, channels)
    pos_emb[:, 0::2] = (pos_idx[:, 0::2] * inv_inc).float().sin()
    pos_emb[:, 1::2] = (pos_idx[:, 1::2] * inv_inc).float().cos()
    return pos_emb.type_as(x.data) + x
    # return Variable(pos_emb.type_as(x.data), requires_grad=False) + x


def add_timing_signal_t(x, t, max_timescale=1e4):
    r"""Adds timing signal at time-step t to x"""
    batch, _, channels = x.size()
    nts = channels // 2
    log_inc = math.log(max_timescale) / nts
    log_inv_inc = -log_inc * torch.arange(0, nts)
    inv_inc = log_inv_inc.exp().view(1, nts)
    pos_emb = torch.FloatTensor(1, channels)
    pos_emb[:, 0::2] = (inv_inc * t).sin()
    pos_emb[:, 1::2] = (inv_inc * t).cos()
    return pos_emb.type_as(x.data) + x
    # return Variable(pos_emb.type_as(x.data), requires_grad=False) + x


def get_padding_mask(q, k):
    r"""Gets padding mask when use query q for key k
    Args:
        q: a Variable LongTensor with shape (batch, length_q)
        k: a Variable LongTensor with shape (batch, length_k)
    Returns:
        a ByteTensor with shape (batch, length_q, length_k)
    """

    masked_pads = k.data.eq(0)
    return masked_pads[:, None, :].expand(k.size(0), q.size(1), k.size(1))


def get_causal_mask(q):
    r"""Gets causal mask. This prevents attention mechanism looks into future
    Args:
        q: a LongTensor with shape (batch, length_q)
    Returns:
        a ByteTensor with shape (batch, length_q, length_q)
    """
    batch, length = q.size()
    tt = torch.cuda if q.is_cuda else torch
    mask = tt.ByteTensor(length, length).fill_(1).triu_(1)
    causal_mask = mask.unsqueeze(0).expand(batch, length, length)
    return causal_mask


class SubLayer(nn.Module):
    r'''A layer consists of one attention layer and a residual feed forward'''
    def __init__(self, input_size, num_heads, head_size, inner_size, dropout):
        super(SubLayer, self).__init__()
        self.attn = modules.MultiHeadAttention(input_size,
                                               num_heads,
                                               head_size)
        self.layer_norm = nn.LayerNorm(input_size)
        self.resff = modules.ResFF(input_size, inner_size, dropout)

    def forward(self, input, mask=None):
        output, attn = self.attn(input, input, input, mask)
        output = self.layer_norm(output + input)
        return self.resff(output), attn


class Encoder(nn.Module):
    r'''Attentive Encoder'''

    def __init__(self, input_size, vocab_size, num_heads, head_size,
                 num_layers, inner_size, dropout):
        super(Encoder, self).__init__()
        self.lut = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.layers = nn.ModuleList(
            [SubLayer(input_size, num_heads, head_size, inner_size, dropout)
             for i in range(num_layers)])

    def forward(self, input):
        mask = get_padding_mask(input, input)
        word_vec = self.lut(input)
        outputs = [add_timing_signal(word_vec, 1000)]
        attns = []
        for i, layer in enumerate(self.layers):
            output, attn = layer(outputs[i], mask)
            attns += [attn]
            outputs += [output]
        self.attns = attns  # dirty hack to expose attention for visualization
        return outputs
class Transformer(nn.Module):
    r'''Fully Attentional Language Model'''

    def __init__(self, input_size, vocab_size, num_heads, head_size,
                 num_layers, inner_size, dropout, tied=False):
        super(Transformer, self).__init__()
        self.lut = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.layers = nn.ModuleList(
            [SubLayer(input_size, num_heads, head_size, inner_size, dropout)
             for i in range(num_layers)])
        self.generator = nn.Linear(input_size, vocab_size)
        if tied:
            self.lut.weight = self.generator.weight

    def forward(self, input, last=False):
        input = input.t()
        mask = get_causal_mask(input)
        word_vec = self.lut(input)
        outputs = add_timing_signal(word_vec, 1000)
        attns = []
        for i, layer in enumerate(self.layers):
            outputs, attn = layer(outputs, mask)
            attns += [attn]
        if last:
            last_outputs = []
            lengths = list(input.data.ne(0).sum(1) - 1)
            b = [i for i in range(len(lengths))]
            last_outputs = outputs[b, lengths, :]
            logits = self.generator(last_outputs)
        else:
            outputs = outputs.transpose(0, 1).contiguous()
            logits = self.generator(outputs.view(-1, outputs.size(-1)))
        return F.log_softmax(logits, dim=-1), attns
