from abc import ABCMeta, abstractmethod
from typing import override
import torch.nn as nn
import torch

hidden_size, num_layers, dropout = 128, 1, 0.1


class Seq2Seq(nn.Module, metaclass=ABCMeta):

    def __init__(self, input_size, output_size, *, max_len, padding_idx=0):
        super().__init__()
        self.max_len, self.pad = max_len, padding_idx
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.emi = nn.Embedding(input_size, hidden_size, padding_idx)
        self.emo = nn.Embedding(output_size, hidden_size, padding_idx)
        self.lin = nn.Linear(hidden_size, output_size)
        self.out = nn.Dropout(dropout)

    def forward(self, inputs, outputs=None, *, bos=1, teacher_forcing=False):
        x, mx = self.out(self.emi(inputs)), inputs == self.pad
        Y, (x, hidden) = [], self.encode(x, mx)
        output = torch.full((inputs.size(0), 1), bos, dtype=int, device=inputs.device)
        for i in range(self.max_len if outputs is None else outputs.size(-1)):
            y, hidden = self.decode(self.out(self.emo(output)), hidden, x, mx)
            y = self.lin(y)
            Y.append(y)
            if outputs is None or not teacher_forcing:
                output = y.argmax(dim=-1).detach()
            else:
                output = outputs[:, i : i + 1]
            if not self.training and torch.all(output == bos):
                break
        return torch.cat(Y, dim=-2)

    def encode(self, x, mx):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        lengths = torch.sum(~mx, dim=-1).cpu()
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, hidden = self.encoder(x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=self.pad)
        return x, hidden

    @abstractmethod
    def decode(self, y, hidden, x, mx): ...


class VanillaSeq2Seq(Seq2Seq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

    @override
    def decode(self, y, hidden, *_):
        return self.decoder(y, hidden)


class PeakySeq2Seq(Seq2Seq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = nn.GRU(
            2 * hidden_size, hidden_size, num_layers, batch_first=True
        )

    @override
    def decode(self, y, hidden, x, _):
        y = torch.concat([y, x[:, -1:, :]], dim=-1)
        return self.decoder(y, hidden)


class AttentionSeq2Seq(Seq2Seq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = nn.GRU(
            2 * hidden_size, hidden_size, num_layers, batch_first=True
        )
        self.attention = Attention(hidden_size)

    @override
    def decode(self, y, hidden, x, mx):
        context = self.attention(hidden[-1].unsqueeze(1), x, mx)
        y = torch.concat([y, context], dim=-1)
        return self.decoder(y, hidden)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.wa = nn.Linear(hidden_size, hidden_size)
        self.ua = nn.Linear(hidden_size, hidden_size)
        self.va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys, mask):
        scores = self.va((self.wa(query) + self.ua(keys)).tanh()).squeeze(-1)
        weights = scores.masked_fill(mask, float("-inf")).unsqueeze(-2).softmax(dim=-1)
        context = weights @ keys
        return context


if __name__ == "__main__":
    from dataset import fra_words, eng_words, seq_len

    input_size, output_size = len(fra_words) + 2, len(eng_words) + 2
    for Modal in (VanillaSeq2Seq, PeakySeq2Seq, AttentionSeq2Seq):
        print(Modal(input_size, output_size, max_len=seq_len))
