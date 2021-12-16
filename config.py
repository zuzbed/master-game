class Config:
    colors = 6
    places = 4
    max_turns = 16
    lstm_nbr = 2
    lstm_size = 512
    feedback_embedding_out = 250
    guess_embedding_out = lstm_size - feedback_embedding_out
    reinforce_alpha = 0.001 # learning rate coefficient
