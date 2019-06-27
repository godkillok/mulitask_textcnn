#!/usr/bin/python3
# -*- coding: utf-8 -*-
from model import Model
from focal_loss import focal_loss_softmax
import tensorflow as tf

def gelu(input_tensor):
     """Gaussian Error Linear Unit.
     This is a smoother version of the RELU.
     Original paper: https://arxiv.org/abs/1606.08415
     Args:
        input_tensor: float Tensor to perform activation.
     Returns:
       `input_tensor` with the GELU activation applied.
     """
     cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
     return input_tensor * cdf


class CnnModel(Model):
    def __init__(self, config, input_ids, author_id, category_ids,  is_training):
        super(CnnModel, self).__init__(config, input_ids, is_training)
        self.author_id = author_id
        self.category_ids = category_ids


    def build_network(self):
        """
        Build network function.
        """
        initializer_range=0.02
        embedding_table = tf.get_variable(name="embedding_table", shape=[self.config['word_dim'], self.config['id_word']],
                                          initializer=self.truncated_normal_initializer(initializer_range),
                                          dtype=tf.float32, trainable=True)

        sentence = tf.nn.embedding_lookup(embedding_table, self.input_ids)
        #self.variable_summaries('embedding', embedding)
        # embedded_words_expanded = self. expand_dims(embedding, -1)
        logits, predict_label_ids, l2_loss = self.build_cnn(sentence)
        return logits, predict_label_ids, l2_loss
    #build_loss(self, labels, logits, l2_loss=0.0)
    def build_loss(self, labels, logits, l2_loss=0):
        """Build loss function.
        args:
          labels: Actual label.
          logits:
        """
        # Loss
        with tf.variable_scope("loss"):
            if self.config['use_focal_loss']:
                losses = focal_loss_softmax(labels=labels, logits=logits)
                loss = losses + self.config['l2_reg_lambda'] * l2_loss
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                loss = tf.reduce_mean(losses) + self.config['l2_reg_lambda']*l2_loss
        return loss

    def build_cnn(self, sentence):
        sentence=tf.expand_dims(sentence, -1)

        pooled_outputs = []
        l2_loss = tf.constant(0.0)  # 先不用，写0

        for filter_size in self.config['filter_sizes']:
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                conv = tf.layers.conv2d(
                    sentence,
                    filters=self.config['num_filters'],
                    kernel_size=[filter_size, self.config['word_dim']],
                    strides=(1, 1),
                    padding="VALID"
                )  # activation=tf.nn.relu
                # conv = tf.layers.batch_normalization(conv, training=(mode == tf.estimator.ModeKeys.TRAIN))
                conv = tf.nn.relu(conv)
                if 'dropout_rate' in self.config and self.config['dropout_rate'] > 0.0:
                    # h_pool_flat = tf.layers.batch_normalization(h_pool_flat, training=(mode == tf.estimator.ModeKeys.TRAIN))

                    conv = tf.layers.dropout(conv, self.config['dropout_rate'],
                                             training=self.is_training)

                pooled = tf.layers.max_pooling2d(
                    conv,
                    pool_size=[self.config['dropout_rate'] - filter_size + 1, 1],
                    strides=(1, 1),
                    padding="VALID")
                pooled_outputs.append(pooled)

        h_pool = tf.concat(pooled_outputs, 3)  # shape: (batch, 1, len(filter_size) * embedding_size, 1)
        h_pool_flat = tf.reshape(h_pool, [-1, self.config['num_filters'] * len(self.config['filter_sizes'])])  # shape: (batch, len(filter_size) * embedding_size)
        if 'dropout_rate' in self.config and self.config['dropout_rate'] > 0.0:
            # h_pool_flat = tf.layers.batch_normalization(h_pool_flat, training=(mode == tf.estimator.ModeKeys.TRAIN))

            h_pool_flat = tf.layers.dropout(h_pool_flat, self.config['dropout_rate'],
                                            training=self.is_training)
        h_pool_flat = tf.layers.batch_normalization(h_pool_flat, training=self.is_training)

        logits = tf.layers.dense(h_pool_flat, self.config['label_size'], activation=None)

        # with tf.variable_scope("output"):
        #     output_w = tf.get_variable("output_w", shape=[hidden_size, self.config['label_size']])
        #     output_b =  self.initialize_bias("output_b", shape=self.config['label_size'])
        #     logits = tf.nn.xw_plus_b(output_layer, output_w, output_b)
        #     l2_loss += tf.nn.l2_loss(output_w) + tf.nn.l2_loss(output_b)

        predict_label_ids = tf.argmax(logits, axis=1, name="predict_label_id")  # 预测结果
        return logits, predict_label_ids, l2_loss

