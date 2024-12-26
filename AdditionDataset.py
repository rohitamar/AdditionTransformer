import torch
from torch.utils.data import Dataset
import random

SOS = 10
EOS = 11
SEP = 12
PAD = 13
BLANK = 14

def split_number(x):
	return [int(c) for c in str(x)]

class AdditionDataset(Dataset):
	
	def __init__(self, n, num_samples):
		all_input, all_output = [], []
		max_len = 3 * n + 6
		A, B = [], []
		for _ in range(num_samples):
			a = random.randint(1, 10 ** n - 1)
			b = random.randint(1, 10 ** n - 1)

			A, B, AB = split_number(a), split_number(b), split_number(a + b)

			input = [SOS] + A + [SEP] + B + [EOS]
			input += [PAD] * (max_len - len(input))

			output = [BLANK] * (2 + len(A) + len(B)) + AB + [EOS]
			output += [PAD] * (max_len - len(output))

			all_input.append(input)
			all_output.append(output)

		self.A = torch.tensor(A)
		self.B = torch.tensor(B)

		self.all_input = torch.tensor(all_input)
		self.all_output = torch.tensor(all_output)

	def __getitem__(self, idx):
		return self.all_input[idx], self.all_output[idx]

	def __len__(self):
		return len(self.all_input)