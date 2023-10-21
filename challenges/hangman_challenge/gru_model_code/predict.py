import numpy as np
import torch

def predict(
    model,
    incomplete_word, 
    guessed_letter_list,
    vocab_size = 26
):
    """
    Parameters
        model (torch.nn.module): loaded trained model 
        incomplete_word (str): word with missing characters Ex: "g*t*"
        guessed_letter_list (list): Guessed letters till now
    """
    char_to_id = {chr(97+x): x for x in range(vocab_size)}
    char_to_id['BLANK'] = vocab_size
    id_to_char = {v:k for k,v in char_to_id.items()}

    # Get miss char list
    correct_guessed_letter_list = list(set([c for c in incomplete_word if(c != "*")]))
    guessed_letter_list = list(set(guessed_letter_list))
    missed_letter_list = [c for c in guessed_letter_list if(c not in correct_guessed_letter_list)]

    #convert string into desired input tensor
    encoded = np.zeros((len(char_to_id)))
    for i, c in enumerate(incomplete_word):
        if(c == '*'):
            encoded[i] = len(id_to_char) - 1 
        else:
            encoded[i] = char_to_id[c]

    inputs = np.array(encoded)[None, :]
    inputs = torch.from_numpy(inputs).long()

    #encode the missed characters
    miss_encoded = np.zeros((len(char_to_id) - 1))
    for c in missed_letter_list:
        miss_encoded[char_to_id[c]] = 1
    miss_encoded = np.array(miss_encoded)[None, :]
    miss_encoded = torch.from_numpy(miss_encoded).float()

    input_lens = np.array([len(incomplete_word)])
    input_lens= torch.from_numpy(input_lens).long()	

    #pass through model
    output = model(inputs, input_lens, miss_encoded).detach().cpu().numpy()[0]

    #sort predictions
    sorted_predictions = np.argsort(output)[::-1]
    
    # Get predicted char
    for c in sorted_predictions:
        c_char = chr(97 + c)
        if(c_char not in guessed_letter_list):
            return c_char
    return "a"