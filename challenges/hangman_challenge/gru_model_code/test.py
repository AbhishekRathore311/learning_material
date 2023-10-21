import torch
import math

from get_current_epoch_data import get_current_epoch_data
from get_current_batch_data import get_current_batch_data

def test(
	epoch,
	model,
	total_epochs,
	encoded_test_word_list,
	batch_size,
	vocab_size,
	cuda
):
	model.eval()

	## Initialize epoch loss
	test_loss = 0.0
	test_miss_penalty = 0.0

	## Without gradient update
	with torch.no_grad():
		## Get cur_epoch_train_data_list
		cur_epoch_test_data_list = get_current_epoch_data(
			encoded_word_list = encoded_test_word_list, 
			epoch_number = epoch,
			total_epochs = total_epochs,
			vocab_size = vocab_size
		)

		## Loop over batches
		no_batches = int(math.ceil(len(cur_epoch_test_data_list) / batch_size))
		for batch_id in range(no_batches):
			## Get batch
			inputs, labels, miss_chars, input_lengths = get_current_batch_data(
				cur_epoch_data_list = cur_epoch_test_data_list, 
				batch_id = batch_id, 
				batch_size = batch_size,
				vocab_size = vocab_size
			)
			
			## Embeddings should be of dtype long
			inputs = torch.from_numpy(inputs).long()
			
			## Convert to torch tensors
			labels = torch.from_numpy(labels).float()
			miss_chars = torch.from_numpy(miss_chars).float()
			input_lengths = torch.from_numpy(input_lengths).long()

			if(cuda==True):
				inputs = inputs.cuda()
				labels = labels.cuda()
				miss_chars = miss_chars.cuda()
				input_lengths = input_lengths.cuda()

			# zero the parameter gradients
			model.optimizer.zero_grad()
			
			# Forward Pass
			outputs = model(inputs, input_lengths, miss_chars)
			loss, miss_penalty = model.calculate_loss(outputs, labels, input_lengths, miss_chars, cuda)
			test_loss += loss.item()
			test_miss_penalty += miss_penalty.item()

	# Average out the losses
	test_loss = (test_loss / no_batches)
	test_miss_penalty = (test_miss_penalty / no_batches)
	return test_loss, test_miss_penalty