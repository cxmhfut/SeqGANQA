import tensorflow as tf

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def cut(resp):
    for time in range(len(resp)):
        if resp[time] == EOS_ID:
            resp = resp[:time + 1]
            break
    return resp


class discriminator_model(object):
    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.lstm_size = config.lstm_size
        self.keep_prob_dropout = config.keep_prob_dropout
        self.num_layer = config.num_layer
        self.max_post_length = config.max_post_length
        self.max_resp_length = config.max_resp_length
        self.max_gradient_norm = config.max_gradient_norm
        self.batch_size = config.batch_size_num
        self.learning_rate = config.learning_rate

        self.post_input = None
        self.post_length = None
        self.resp_input = None
        self.resp_length = None
        self.labels = None
        self.loss = None
        self.poss = None
        self.accuracy = None
        self.train_op = None

        self.build_model()

    def create_rnn_cell(self):
        def single_rnn_cell():
            single_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(single_cell,
                                                 input_keep_prob=1.0,
                                                 output_keep_prob=self.keep_prob_dropout)
            return cell

        cells = tf.nn.rnn_cell.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layer)])
        return cells

    def concat(self, lstm_tuple):
        return tf.concat([tf.concat([pair.c, pair.h], axis=1) for pair in lstm_tuple], axis=1)

    def build_model(self):
        with tf.variable_scope('d_model'):
            self.post_input = tf.placeholder(tf.int32, [self.max_post_length, None], name='post_input')
            self.post_length = tf.placeholder(tf.int32, [None], name='post_length')

            self.resp_input = tf.placeholder(tf.int32, [self.max_resp_length, None], name='resp_input')
            self.resp_length = tf.placeholder(tf.int32, [None], name='resp_length')

            self.labels = tf.placeholder(tf.int64, [None])

            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size], dtype=tf.float32)

            post_embedded = tf.nn.embedding_lookup(embedding, self.post_input)
            resp_embedded = tf.nn.embedding_lookup(embedding, self.resp_input)

            with tf.variable_scope('encoder'):
                cell = self.create_rnn_cell()
                post_output, post_state = tf.nn.dynamic_rnn(cell,
                                                            post_embedded,
                                                            sequence_length=self.post_length,
                                                            time_major=True,
                                                            dtype=tf.float32)

            with tf.variable_scope('encoder', reuse=True):
                resp_output, resp_state = tf.nn.dynamic_rnn(cell,
                                                            resp_embedded,
                                                            sequence_length=self.resp_length,
                                                            time_major=True,
                                                            dtype=tf.float32)

            post_state_concat = self.concat(post_state)
            resp_state_concat = self.concat(resp_state)

            state_final_concat = tf.concat([post_state_concat, resp_state_concat], axis=1)

            logits = tf.layers.dense(state_final_concat, 2)
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
            self.poss = tf.nn.softmax(logits)[:, 1]

            result = tf.argmax(logits, axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(result, self.labels), tf.float32))

            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, params))

    def update(self, sess, generator, reader):
        batch = reader.get_batch(self.batch_size)

        resp_genarate = generator.generate(sess, batch, 'sample')
        feed_post = [[] for _ in range(self.max_post_length)]
        feed_resp = [[] for _ in range(self.max_resp_length)]
        feed_post_length = []
        feed_resp_length = []
        feed_labels = []

        for post, resp in batch:
            feed_post_length.append(len(post))
            feed_resp_length.append(len(resp))
            feed_labels.append(1)
            for time in range(self.max_post_length):
                feed_post[time].append(post[time] if time < len(post) else PAD_ID)
                feed_resp[time].append(resp[time] if time < len(resp) else PAD_ID)

        for index in range(self.batch_size):
            post = batch[index][0]
            resp = resp_genarate[index]
            resp = cut(resp)
            feed_post_length.append(len(post))
            feed_resp_length.append(len(resp))
            feed_labels.append(0)
            for time in range(self.max_post_length):
                feed_post[time].append(post[time] if time < len(post) else PAD_ID)
                feed_resp[time].append(resp[time] if time < len(resp) else PAD_ID)

        feed_dict = {}
        feed_dict[self.post_input] = feed_post
        feed_dict[self.resp_input] = feed_resp
        feed_dict[self.post_length] = feed_post_length
        feed_dict[self.resp_length] = feed_resp_length
        feed_dict[self.labels] = feed_labels

        poss, loss, accuracy, _ = sess.run([self.poss, self.loss, self.accuracy, self.train_op], feed_dict=feed_dict)
        print('Discriminator- Loss:', loss, ' Accuracy:', accuracy)

    def evaluate(self, sess, batch, reader):
        feed_post = [[] for _ in range(self.max_post_length)]
        feed_resp = [[] for _ in range(self.max_resp_length)]
        feed_post_length = []
        feed_resp_length = []
        for post, resp in batch:
            feed_post_length.append(len(post))
            resp = cut(resp)
            feed_resp_length.append(len(resp))
            for time in range(self.max_post_length):
                feed_post[time].append(post[time] if time < len(post) else PAD_ID)
                feed_resp[time].append(resp[time] if time < len(resp) else PAD_ID)

        feed_dict = {}
        feed_dict[self.post_input] = feed_post
        feed_dict[self.resp_input] = feed_resp
        feed_dict[self.post_length] = feed_post_length
        feed_dict[self.resp_length] = feed_resp_length

        poss = sess.run(self.poss, feed_dict=feed_dict)
        return poss
