from M5 import M5
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.nn.utils.rnn import pad_sequence


a = torch.ones(1, 3)
b = torch.ones(1, 3)
c = torch.ones(1, 3)
x = pad_sequence([a, b, c])
print(x)



def collate_fn(file):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors = []

    # Gather in lists, and encode labels as indices
    waveform, sample_rate = torchaudio.load(file)
    tensors += [waveform]
    print(tensors)

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)

    return tensors

if __name__=="__main__":
    collate_fn("Mortimer_Wave/S2R6_2_CS.wav")
