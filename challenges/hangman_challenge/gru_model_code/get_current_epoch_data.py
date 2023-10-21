import numpy as np

def get_current_epoch_data(
	encoded_word_list, 
	epoch_number, 
	total_epochs,
	vocab_size = 26
):
	## As training progresses the prob of dropping chars increases using sigmoid on epoch
	drop_char_probability = 1/(1+np.exp(-epoch_number/total_epochs))
	cur_epoch_data_list = []
	all_character_set = set([chr(97+x) for x in range(vocab_size)])
	char_to_id = {chr(97+x): x for x in range(vocab_size)}
	char_to_id['BLANK'] = vocab_size

	for i, (encoded_word, char_location_list, word_set) in enumerate(encoded_word_list):
		## Number of characters to drop
		num_char_to_drop = np.random.binomial(len(char_location_list), drop_char_probability)
		if num_char_to_drop == 0:
			num_char_to_drop = 1

		## Drop chars inversely proportional to number of occurences of each character
		## For Ex: goto, char_location_list = [[0], [1, 3], [2]]
		## drop_char_probability_list = [0.4, 0.2, 0.4]
		## to_drop = [0, 1]
		drop_char_probability_list = [1/len(x) for x in char_location_list]
		drop_char_probability_list = [x/sum(drop_char_probability_list) for x in drop_char_probability_list]
		to_drop = np.random.choice(len(char_location_list), num_char_to_drop, p=drop_char_probability_list, replace=False)

		## Cha positions to drop
		## For Ex: goto, char_location_list = [[0], [1, 3], [2]] and to_drop = [0, 1]
		## drop_char_idx = [0, 1, 3]
		drop_char_idx = []
		for char_group in to_drop:
			drop_char_idx += char_location_list[char_group]
		
		## drop_char_idx = model target
		## Assuming voab_size = 4
		## For Ex: goto, encoded_word = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
		## unclipped_target = [1, 0, 2, 0]
		## target = [1, 0, 1, 0]
		unclipped_target = np.sum(encoded_word[drop_char_idx], axis=0)
		target = np.clip(unclipped_target, 0, 1)

		## Remove blank in target
		target = target[:-1]
		
		## Drop chars and assign blank_character
		input_vec = np.copy(encoded_word)
		blank_vec = np.zeros((1, vocab_size + 1))
		blank_vec[0, vocab_size] = 1
		input_vec[drop_char_idx] = blank_vec

		## Provide character id instead of 1-hot encoded vector for embedding
		input_vec = np.argmax(input_vec, axis=1)
		## For Ex: goto, encoded_word = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
		## drop_char_idx = [0, 1, 3]
		## target = [1, 0, 1, 0]
		## input_vec = [26, 26, 19, 26] (26 = BLANK, 19 = t)
		
		## randomly pick a few characters from vocabulary as characters which were predicted but declared as not present by game
		not_present_char_sorted_array = np.array(sorted(list(all_character_set - word_set)))
		num_missed_chars = np.random.randint(0, 10)
		miss_char_sorted_array = np.random.choice(not_present_char_sorted_array, num_missed_chars)
		miss_char_id_sorted_list = [char_to_id[x] for x in miss_char_sorted_array]
		## Ex word is 'goto', num_missed_chars = 2, miss_char_id_sorted_list = [1, 3] 
		## (which correspond to the characters b and d)
		
		miss_vec = np.zeros(vocab_size)
		miss_vec[miss_char_id_sorted_list] = 1
		## If vocab_size = 6, b = 1, d = 3 and b, d are missed
		## miss_vec = [0, 1, 0, 1, 0, 0]
		
		## Append tuple to cur_epoch_data_list
		cur_epoch_data_list.append((input_vec, target, miss_vec))

	## Shuffle dataset before feeding batches to the model
	np.random.shuffle(cur_epoch_data_list)
	return cur_epoch_data_list