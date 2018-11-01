import tensorflow as tf
from tensorflow.contrib import seq2seq

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


class generator_model(object):
    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.lstm_size = config.lstm_size
        self.keep_prob_dropout = config.keep_prob_dropout
        self.num_layer = config.num_layer
        self.max_length_encoder = config.max_length_encoder
        self.max_length_decoder = config.max_length_decoder
        self.max_gradient_norm = config.max_gradient_norm
        self.batch_size = config.batch_size_num
        self.learning_rate = config.learning_rate
        self.beam_width = config.beam_width

        self.encoder_inputs = None
        self.encoder_inputs_length = None
        self.decoder_targets = None
        self.decoder_targets_length = None
        self.max_target_sequence_length = None
        self.mask = None
        self.reward = None
        self.start_tokens = None
        self.max_inference_length = None
        self.decoder_logits_train = None
        self.decoder_predict_train = None
        self.loss_pretrain = None
        self.summary_op = None

    def create_rnn_cell(self):
        def single_rnn_cell():
            single_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(single_cell,
                                                 input_keep_prob=1.0,
                                                 output_keep_prob=self.keep_prob_dropout)
            return cell

        cells = tf.nn.rnn_cell.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layer)])
        return cells

    def build_model(self):
        """
        build model
        :return:
        """
        with tf.variable_scope('g_model'):
            # 1 定义模型的placeholder
            # encoder
            self.encoder_inputs = tf.placeholder(tf.int32, [self.max_length_encoder, None], name='encoder_inputs')
            self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
            # decoder
            self.decoder_targets = tf.placeholder(tf.int32, [self.max_length_decoder, None], name='decoder_targets')
            self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
            self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
            self.mask = tf.sequence_mask(self.decoder_targets_length,
                                         self.max_target_sequence_length,
                                         dtype=tf.float32,
                                         name='masks')

            # for updating
            self.reward = tf.placeholder(tf.float32, [self.max_length_decoder, None], name='reward')
            self.start_tokens = tf.placeholder(tf.int32, [None], name='start_tokens')  # for partial-sampling
            self.max_inference_length = tf.placeholder(tf.int32, [None], name='max_inference_length')  # for inference

            # 2 定义模型的encoder部分
            with tf.variable_scope('encoder'):
                encoder_cell = self.create_rnn_cell()
                embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
                encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                                   encoder_inputs_embedded,
                                                                   sequence_length=self.encoder_inputs_length,
                                                                   dtype=tf.float32)

            # 3 定义模型的decoder部分
            with tf.variable_scope('decoder'):
                encoder_inputs_length = self.encoder_inputs_length
                # 定义要使用的attention机制
                attention_mechanism = seq2seq.BahdanauAttention(num_units=self.lstm_size,
                                                                memory=encoder_outputs,
                                                                memory_sequence_length=encoder_inputs_length)
                decoder_cell = self.create_rnn_cell()
                decoder_cell = seq2seq.AttentionWrapper(cell=decoder_cell,
                                                        attention_mechanism=attention_mechanism,
                                                        attention_layer_size=self.lstm_size,
                                                        name='Attention_Wrapper')
                # 定义decoder阶段的初始状态，直接使用encoder阶段的最后一个隐层状态进行赋值
                decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone(
                    cell_state=encoder_state)
                output_layer = tf.layers.Dense(self.vocab_size,
                                               kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

                ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
                decoder_inputs = tf.concat([tf.fill([self.batch_size, 1], tf.cast(GO_ID, dtype=tf.int32)), ending], 1)
                decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, decoder_inputs)

                # train
                helper_train = seq2seq.TrainingHelper(decoder_inputs_embedded,
                                                      self.decoder_targets_length,
                                                      time_major=True)
                decoder_train = seq2seq.BasicDecoder(decoder_cell,
                                                     helper_train,
                                                     decoder_initial_state,
                                                     output_layer=output_layer)
                decoder_output_train, decoder_state_train, _ = seq2seq.dynamic_decode(decoder_train,
                                                                                      swap_memory=True,
                                                                                      output_time_major=True,
                                                                                      impute_finished=True,
                                                                                      maximum_iterations=self.decoder_targets_length)
                self.decoder_logits_train = tf.identity(decoder_output_train.rnn_output)
                self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')
                self.loss_pretrain = seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                           targets=self.decoder_targets,
                                                           weights=self.mask)
                tf.summary.scalar('loss', self.loss_pretrain)
                self.summary_op = tf.summary.merge_all()
