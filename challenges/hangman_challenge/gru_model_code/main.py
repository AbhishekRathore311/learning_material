from train import train

train(
	total_epochs = 10,
	encoded_train_words_location = "../data/gru_model/encoded_test_words_fraction.pickle",
	encoded_test_words_location = "../data/gru_model/encoded_test_words_fraction.pickle",
	batch_size = 2000,
	vocab_size = 26,
	cuda = False,
    reset_after = 10,
	save_every = 10,
	model_output_location = "model_output",
	gru_hidden_dim = 512,
	gru_num_layers = 2,
	char_embedding_dim = 128,
	missed_char_linear_dim = 256,
	nn_hidden_dim = 256,
	gru_dropout = 0.3,
	learning_rate = 0.0005
)