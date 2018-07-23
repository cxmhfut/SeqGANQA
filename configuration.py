
class generation_config(object):
    """Wrapper class for generator hyperparameter"""

    def __init__(self):
        self.vocab_size = None
        self.embedding_size = None
        self.lstm_size = None
        self.num_layer = None
        self.max_length_encoder = None
        self.max_length_decoder = None
        self.max_gradient_norm = None
        self.batch_size_num = None
        self.learning_rate = None
        self.beam_width = None
        self.embed = None