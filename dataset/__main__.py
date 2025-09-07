import torch
from . import fra_eng, fra_words, eng_words, seq_len, get_dataset

torch.manual_seed(9527)
print(len(fra_eng), len(fra_words) + 2, len(eng_words) + 2, seq_len)
for train, test in zip(*get_dataset()):
    print(*map(lambda x: x.shape, train))
    print(*test, sep="\n")
    break
