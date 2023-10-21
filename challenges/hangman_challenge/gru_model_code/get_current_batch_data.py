import numpy as np

def batchify_words(input_vec_list, vocab_size):
    total_seq = len(input_vec_list)
    max_len = max([len(x) for x in input_vec_list])
    batched_input_list = []

    for word in input_vec_list:
        if max_len != len(word):
            ## Add blanks to get max len
            blank_vec = (vocab_size * np.ones((max_len - word.shape[0])))
            word = np.concatenate((word, blank_vec), axis=0)
        batched_input_list.append(word)

    return np.array(batched_input_list)

def get_current_batch_data(
    cur_epoch_data_list, 
    batch_id, 
    batch_size,
    vocab_size
):
    if(((batch_id + 1) * batch_size) <= len(cur_epoch_data_list)):
        start_index = (batch_id * batch_size)
        end_index = ((batch_id + 1) * batch_size)
        cur_batch_data_list = cur_epoch_data_list[start_index: end_index]
    else:
        start_index = (batch_id * batch_size)
        end_index = len(cur_epoch_data_list)
        cur_batch_data_list = cur_epoch_data_list[start_index: end_index]
    
    ## Convert to numpy arrays
    word_length_array = np.array([len(x[0]) for x in cur_batch_data_list])
    input_vec_list = [x[0] for x in cur_batch_data_list]
    batched_input_array = batchify_words(input_vec_list, vocab_size)
    batched_label_array = np.array([x[1] for x in cur_batch_data_list])
    batched_missed_char_array = np.array([x[2] for x in cur_batch_data_list])

    ## Return batch
    return batched_input_array, batched_label_array, batched_missed_char_array, word_length_array