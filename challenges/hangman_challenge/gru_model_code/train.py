import math
import pickle
import torch
import pandas as pd
from tqdm import tqdm

from sequence_model import HangmanGRU
from get_current_epoch_data import get_current_epoch_data
from get_current_batch_data import get_current_batch_data
from test import test

def train(
    total_epochs = 3000,
    encoded_train_words_location = None,
    encoded_test_words_location = None,
    batch_size = 4000,
    vocab_size = 26,
    cuda = False,
    reset_after = 300,
    save_every = 300,
    model_output_location = None,
    gru_hidden_dim = 512,
    gru_num_layers = 2,
    char_embedding_dim = 128,
    missed_char_linear_dim = 256,
    nn_hidden_dim = 256,
    gru_dropout = 0.3,
    learning_rate = 0.0005
):
    ## Load model and set it to train mode
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
    model.train()

    ## Get encoded_train_word_list
    encoded_train_word_list = pickle.load(open(encoded_train_words_location, "rb"))
    
    ## Get encoded_test_word_list
    encoded_test_word_list = pickle.load(open(encoded_test_words_location, "rb"))

    ## Lists to store losses
    train_loss_list = []
    train_miss_penalty_list = []
    test_loss_list = []
    test_miss_penalty_list = []
    epoch_list = []

    ## Loop over Train Data
    for epoch in tqdm(range(1, total_epochs+1)):
        ## Initialize epoch loss
        train_loss = 0.0
        train_miss_penalty = 0.0

        ## Get cur_epoch_train_data_list
        if(((epoch - 1) % reset_after) == 0):
            cur_epoch_train_data_list = get_current_epoch_data(
                encoded_word_list = encoded_train_word_list, 
                epoch_number = epoch, 
                total_epochs = total_epochs,
                vocab_size = vocab_size
            )

        ## Loop over batches
        no_batches = int(math.ceil(len(cur_epoch_train_data_list) / batch_size))
        for batch_id in range(no_batches):
            ## Get batch
            inputs, labels, miss_chars, input_lengths = get_current_batch_data(
                cur_epoch_data_list = cur_epoch_train_data_list, 
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

            ## Zero the parameter gradients
            model.optimizer.zero_grad()
            
            ## Forward Pass, Loss calculation, Backward Pass, Optimize
            outputs = model(inputs, input_lengths, miss_chars)
            loss, miss_penalty = model.calculate_loss(outputs, labels, input_lengths, miss_chars, cuda)
            loss.backward()
            model.optimizer.step()

            ## store loss
            train_loss += loss.item()
            train_miss_penalty += miss_penalty.item()

            ## Print Info
            print(f"Epoch: {epoch} | Batch: {batch_id} | Input Length: {len(inputs)}")

        # Test model after epoch
        test_loss, test_miss_penalty = test(
            epoch = epoch,
            model = model,
            total_epochs = total_epochs,
            encoded_test_word_list = encoded_test_word_list,
            batch_size = batch_size,
            vocab_size = vocab_size,
            cuda = cuda
        )
        model.train()

        # Store losses
        epoch_list.append(epoch)
        train_loss = (train_loss / no_batches)
        train_loss_list.append(train_loss)
        train_miss_penalty = (train_miss_penalty/ no_batches)
        train_miss_penalty_list.append(train_miss_penalty)
        test_loss_list.append(test_loss)
        test_miss_penalty_list.append(test_miss_penalty)

        # Save Losses
        df_loss = pd.DataFrame(
            {
                "epoch": epoch_list,
                "train_loss": train_loss_list,
                "train_miss_penalty": train_miss_penalty_list,
                "test_loss": test_loss_list,
                "test_miss_penalty": test_miss_penalty_list
            }
        )
        df_loss_location = f"{model_output_location}/df_loss.csv"
        df_loss.to_csv(df_loss_location, index=False)

        # Save model
        if(epoch % save_every == 0):
            model_path = f"{model_output_location}/models"
            model_file_name = f"{model_path}/model_epoch_{str(epoch).zfill(4)}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, model_file_name)