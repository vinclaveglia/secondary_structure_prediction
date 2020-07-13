import torch
import os
from secStructPredictor import get_model, predictSecStructure
import sys

def get_struct_types():
	struct_types = "hst-"
	return struct_types

def get_model_architecture():
	input_dim = 20
	output_dim = 4
	rnn = get_model(input_dim, output_dim, hidden_size=20, n_layers=3)
	return rnn

def get_model_configuration():
	device="cpu"
	model_file = os.path.join(os.getcwd(),'struc_classifier_SD2_0.7907')
	state = torch.load(model_file, map_location=device)
	return state


def main():
	if len(sys.argv) <= 1:
		aa_sequence = "ACDEFGHIKLMNPQRSTVWY"
	else:
		aa_sequence = sys.argv[1].upper()

	print(aa_sequence)

	struct_types = get_struct_types()

	rnn = get_model_architecture()
	configuration = get_model_configuration()
	rnn.load_state_dict(configuration)

	out_idxs = predictSecStructure(rnn, aa_sequence)
	predicted = [struct_types[j] for j in out_idxs]
	print("aminoacids : ",list(aa_sequence))
	print("sec. struct: ", predicted)


if __name__ == '__main__':
    main()