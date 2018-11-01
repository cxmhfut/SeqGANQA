class generator_config(object):
    """Wrapper class for generator hyper parameter"""

    def __init__(self):
        self.vocab_size = 35804
        self.embedding_size = 128
        self.lstm_size = 128
        self.keep_prob_dropout = 0.8
        self.num_layer = 4
        self.max_length_encoder = 40
        self.max_length_decoder = 40
        self.max_gradient_norm = 2
        self.batch_size_num = 20
        self.learning_rate = 0.001
        self.beam_width = 5


class discriminator_config(object):
    """Wrapper class for discriminator hyper parameter"""

    def __init__(self):
        self.vocab_size = 35804
        self.embedding_size = 128
        self.lstm_size = 128
        self.keep_prob_dropout = 0.8
        self.num_layer = 4
        self.max_post_length = 40
        self.max_resp_length = 40
        self.max_gradient_norm = 2
        self.batch_size_num = 20
        self.learning_rate = 0.001
