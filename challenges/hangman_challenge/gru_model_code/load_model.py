import torch
from sequence_model import HangmanGRU

def load_model(
    saved_model_output_location,
    vocab_size = 26,
    cuda = False,
    gru_hidden_dim = 512,
    gru_num_layers = 2,
    char_embedding_dim = 128,
    missed_char_linear_dim = 256,
    nn_hidden_dim = 256,
    gru_dropout = 0.3,
    learning_rate = 0.0005
):
    saved_model_output = torch.load(saved_model_output_location)
    model = HangmanGRU(
        vocab_size = vocab_size,
        gru_hidden_dim = gru_hidden_dim,
        gru_num_layers = gru_num_layers,
        char_embedding_dim = char_embedding_dim,
        missed_char_linear_dim = missed_char_linear_dim,
        nn_hidden_dim = nn_hidden_dim,
        gru_dropout = gru_dropout,
        learning_rate = learning_rate
    )
    model.load_state_dict(saved_model_output["model_state_dict"])
    return model