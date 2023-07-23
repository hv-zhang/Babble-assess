from M5 import M5
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio


def classify(file):
    waveform, sample_rate = torchaudio.load(file)
    model = M5(n_input = waveform.shape[0], n_output=2)
    model.load_state_dict(torch.load("trained_model.ckpt"))
    model.eval()
    waveform = waveform.unsqueeze(0)
    return torch.exp(model(waveform))[0,0,1].item()

def archiveclassify(file):

    # Read in the audio file, pass it through the model
    waveform, sample_rate = torchaudio.load(file)
    print(f'Waveform: {waveform.shape}')
    print(f'Sample Rate: {sample_rate}')

    ## For the waveform, we downsample the audio for faster processing without
    ## losing too much of the classification power.
    #new_sample_rate = 44100 #?????? not sure what is the best
    #transform = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                               #new_freq=new_sample_rate)
    #transformed = transform(waveform)

    ## Pad the wav file to be a certain length.
    def pad_sequence(batch):
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)
    waveform = pad_sequence(waveform)

    print("about to load model!!!!")
    # Load the model from file
    model = M5(n_input = waveform.shape[0], n_output=2)
    model.load_state_dict(torch.load("trained_model.ckpt"))
    model.eval()

    return model(waveform)
    #return "4"

if __name__ == "__main__":
    #pred = testclassify("Mortimer_Wave/synth_Mortimer_1.wav")
    #print(pred)
    pred = classify("Mortimer_Wave/synth_Mortimer_1.wav")
    print(pred)

# -> Change to the correct Python installation (3.8) 
#
# pyenv("Version", "/Library/Frameworks/Python.framework/Versions/3.8/bin/python3")


# -> Import module and then call function
#
# py.importlib.import_module('print')
# py.print.printHello()


# -> If you modified the Python file, here is how to reload it in Matlab.
#
# clear classes
# mod = py.importlib.import_module('hello');
# py.importlib.reload(mod);
# py.print.printHello()


# -> To check version of python:
#
# pe = pyenv;
# pe.Version
# (should return 3.8)
