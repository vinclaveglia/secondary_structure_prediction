import torch
import torch.nn as nn
import os

class RecurrentModel(nn.Module):

    def __init__(self, in_size, out_size, hidden_size, n_layers):
        super().__init__()
        #self.rnn_model = nn.GRU(input_size=in_size, hidden_size=hidden_size, num_layers=n_layers)
        self.rnn_model = nn.RNN(input_size=in_size, hidden_size=hidden_size, num_layers=n_layers)

        self.final_layer = nn.Linear(in_features=hidden_size, out_features=out_size)


    def forward(self, x):

        x, hiddens = self.rnn_model(x)
        #x = torch.relu(x)
        x = self.final_layer(x)
        #x = torch.sigmoid(x)
        return x, hiddens


def get_model(in_size, out_size, hidden_size, n_layers):
    if hidden_size is None:
        hidden_size = in_size
    model = RecurrentModel(in_size, out_size, hidden_size, n_layers)
    print(model)
    return model


def sequenceToTensor(sequence):
    aminoacids = "ACDEFGHIKLMNPQRSTVWY"
    seqTensor = torch.zeros((len(sequence), len(aminoacids)))
    for j, aa in enumerate(sequence):
        aa_idx = aminoacids.index(aa)
        seqTensor[j, aa_idx] = 1.
    return seqTensor


def predictSecStructure(classifier, aa_sequence):
	tensor_seq = sequenceToTensor(aa_sequence)
	with torch.no_grad():
		classifier.eval()
		out, _ = classifier(tensor_seq.unsqueeze(1))
	out_idxs = out.squeeze().argmax(1)
	return out_idxs

