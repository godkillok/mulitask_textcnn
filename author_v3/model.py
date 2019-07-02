#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class Model(object):
    def __init__(self, config, input_ids, is_training):
        """Constructor for Model.
        Args:
          config: Parameter configuration.
          input_ids: A tensor with 2 or more dimensions.
          is_training: True for training model, false for eval model. Controls whether dropout will be applied.
        """
        self.config = config
        self.input_ids = input_ids
        self.is_training = is_training

    def build_network(self):
        """Must defined in subclass.
        """
        raise NotImplementedError("build_network: not implemented!")

    def build_loss(self, labels, logits, l2_loss=0.0):
        """Must defined in subclass.
        """
        raise NotImplementedError("build_loss: not implemented!")

    def initialize_weight(self, name, shape):
        """initialize variable weight.
        Args:
            name: The name of the new or existing variable.
            shape: Shape of the new or existing variable.
        """
        return tf.get_variable(name, shape, dtype=tf.float32, initializer=self.truncated_normal_initializer())

    def initialize_bias(self, name, shape):
        """initialize variable bias.
        Args:
            name: The name of the new or existing variable.
            shape: Shape of the new or existing variable.
        """
        return tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.constant_initializer(0.02))

    def get_activation(self, activation_string):
        """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.
        Args:
          activation_string: String name of the activation function.
        """
        act = activation_string.lower()
        if act == "linear":
            return None
        elif act == "relu":
            return tf.nn.relu
        elif act == "tanh":
            return tf.tanh
        else:
            raise ValueError("Unsupported activation: %s" % act)

    def dropout(self, input_tensor, dropout_prob):
        """Perform dropout.
        Args:
          input_tensor: A tensor with 2 or more dimensions.
          dropout_prob: Python float. The probability of dropping out a value (NOT of
            *keeping* a dimension as in `tf.nn.dropout`).
        """
        if dropout_prob is None or dropout_prob == 0.0:
            return input_tensor
        output = tf.layers.dropout(input_tensor, dropout_prob, training=self.is_training)
        return output

    def layer_norm(self, input_tensor, name=None):
        """Run layer normalization on the last dimension of the tensor.
        Args:
          input_tensor: A tensor with 2 or more dimensions.
        """
        return tf.contrib.layers.layer_norm(
            inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

    def batch_norm(self, input_tensor):
        """Add a batch normalization layer.
        Args:
          input_tensor: A tensor with 2 or more dimensions.
        """
        inputs = tf.layers.batch_normalization(input_tensor, training=self.is_training)
        return inputs

    def attention(self, input_tensor, num_units, attention_length, cell_type):
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units, memory=input_tensor,
                memory_sequence_length=attention_length, normalize=True)
        #Selecting the Cell Type to use
        if cell_type.upper() == 'LSTM':
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
        elif cell_type.upper() == 'GRU':
            cell = tf.contrib.rnn.GRUCell(num_units=num_units)
        else:
            raise ValueError("Error in type of Cell Type provided %s " % cell_type)

        #Wrapping attention to the cell
        atten_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=num_units)
        return atten_cell

    def add_full_connect_layer(self, input_tensor, output_unit):
        """Add a fully connected layer.
        Args:
          input_tensor: A tensor with 2 or more dimensions.
          output_unit: Dimensionality of the output space.
        """
        return tf.layers.dense(input_tensor,output_unit)

    def truncated_normal_initializer(self, initializer_range=0.02):
        """Creates a `truncated_normal_initializer` with the given range.
        Args:
          initializer_range: Standard deviation of the random values to generate.
        """
        return tf.truncated_normal_initializer(stddev=initializer_range)

    def he_initializer(self):
        """Creates a 'he_initializer'."""
        return tf.keras.initializers.he_normal()

    def xavier_initializer(self):
        """Creates a 'xavier_initializer'."""
        return tf.contrib.layers.xavier_initializer()

    def expand_dims(self, input_tensor, axis=-1):
        """Inserts a dimension of 1 into a tensor's shape.
        Args:
          input_tensor: A tensor with 2 or more dimensions.
          axis: Specifies the dimension index at which to expand the shape of `input`.
        """
        return tf.expand_dims(input_tensor, axis)

    def variable_summaries(self, name, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name + '_summaries'):
            name = name.replace(':', '_')
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))

    def load_word2vec(self, emb_path, id_word, embedding_size, initializer_range):
        """Generate word embedding from pre-trained files or randomly.
        Args:
          emb_path: Pre-trained embedding file.
          id_word: A dictionary maps id to word
          embedding_size: Width of the word embeddings.
          axis: Specifies the dimension index at which to expand the shape of `input`.
        """
        n_words = len(id_word)
        embedding_table = np.random.normal(loc=0.0, scale=initializer_range, size=(n_words, embedding_size))
        print('Loading pretrained embeddings from {}...'.format(emb_path))
        pre_trained = {}
        emb_invalid = 0
        for i, line in enumerate(open(emb_path, 'r', encoding='utf8')):
            line = line.rstrip().split()
            if len(line) == embedding_size + 1:
                pre_trained[line[0]] = np.array(
                    [float(x) for x in line[1:]]
                ).astype(np.float32)
            else:
                emb_invalid += 1
        for i in range(n_words):
            word = id_word[str(i)]
            if word in pre_trained:
                embedding_table[i] = pre_trained[word]
        return embedding_table

    def embedding_lookup(self, input_ids, id_word, embedding_size=128, initializer_range=0.02,
                         word_embedding_name="embedding_table", emb_path=None):
        """Looks up words embeddings for id tensor.
        Args:
          input_ids: Tensor of shape [batch_size, max_length] containing word ids.
          id_word: A dictionary maps id to word
          embedding_size: Width of the word embeddings.
          initializer_range: Embedding initialization range.
          word_embedding_name: Name of the embedding table.
        """

        # This function assumes that the input is of shape [batch_size, max_length].
        vocab_size = len(id_word)
        if emb_path is None:
            embedding_table = tf.get_variable(name=word_embedding_name, shape=[vocab_size, embedding_size],
                                              initializer=self.truncated_normal_initializer(initializer_range), dtype=tf.float32,trainable=True)
        else:
            embedding_table = self.load_word2vec(emb_path, id_word, embedding_size, initializer_range)
        output = tf.cast(tf.nn.embedding_lookup(embedding_table, input_ids), tf.float32)
        return output
