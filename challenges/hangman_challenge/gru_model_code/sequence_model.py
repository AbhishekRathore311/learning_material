from torch import nn
from torch.nn import Embedding, Linear, ReLU, GRU
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence
import torch

class HangmanGRU(nn.Module):
    def __init__(
        self, 
        vocab_size = 26,
        gru_hidden_dim = 512,
        gru_num_layers = 2,
        char_embedding_dim = 128,
        missed_char_linear_dim = 256,
        nn_hidden_dim = 256,
        gru_dropout = 0.3,
        learning_rate = 0.0005
    ):
        super(HangmanGRU, self).__init__()

        ## Different model dimentions
        self.gru_hidden_dim = gru_hidden_dim 
        self.gru_num_layers = gru_num_layers

        ## Embedding layer for character input
        self.embedding = Embedding(vocab_size + 1, char_embedding_dim)

        ## Declare GRU
        self.hangman_gru = GRU(
            input_size = char_embedding_dim,
            hidden_size = self.gru_hidden_dim,
            num_layers = self.gru_num_layers,
            dropout = gru_dropout,
            bidirectional=True,
            batch_first=True
        )

        ## Missed characters linear layer
        self.missed_characters_linear_layer = Linear(vocab_size, missed_char_linear_dim)
            
        # NN after GRU output
        nn_in_features = missed_char_linear_dim + (self.gru_hidden_dim * 2)
        self.nn_hidden_layer = Linear(nn_in_features, nn_hidden_dim)
        self.relu = ReLU()
        self.nn_output_layer = Linear(nn_hidden_dim, vocab_size)

        ## Set up optimizer
        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, x_length, missed_characters):
        x = self.embedding(x)
        batch_size, seq_len, _ = x.size()
        x = pack_padded_sequence(x, x_length, batch_first=True, enforce_sorted=False)
        
        ## Run through GRU
        output, hidden = self.hangman_gru(x)
        hidden = hidden.view(self.gru_num_layers, 2, -1, self.gru_hidden_dim)
        hidden = hidden[-1]
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.contiguous().view(hidden.shape[0], -1)

        ## Project missed_characters to higher dimension
        missed_characters = self.missed_characters_linear_layer(missed_characters)
        
        ## Concatenate GRU output and missed_characters
        concatenated = torch.cat((hidden, missed_characters), dim=1)
        
        ## Run NN after GRU
        nn_output = self.nn_hidden_layer(concatenated)
        nn_output = self.relu(nn_output)
        nn_output = self.nn_output_layer(nn_output)
        return nn_output

    def calculate_loss(self, model_out, labels, input_lengths, missed_characters, use_cuda):
        outputs = nn.functional.log_softmax(model_out, dim=1)
        
        ## Calculate model output loss for miss characters
        miss_penalty = torch.sum((outputs * missed_characters), dim=(0,1))/outputs.shape[0]
        
        ## Convert input lengths to float
        input_lengths = input_lengths.float()
        
        ## Weights per example is inversely proportional to length of word
        ## This is because shorter words are harder to predict due to higher chances of missing a character
        weights_orig = (1/input_lengths)/torch.sum(1/input_lengths).unsqueeze(-1)
        weights = torch.zeros((weights_orig.shape[0], 1))    
        
        ## Resize so that torch can process it correctly
        weights[:, 0] = weights_orig

        if use_cuda:
            weights = weights.cuda()
        
        ## Actual Loss
        loss_function = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
        actual_penalty = loss_function(model_out, labels)
        return actual_penalty, miss_penalty