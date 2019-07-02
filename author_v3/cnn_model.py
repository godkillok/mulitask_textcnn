#!/usr/bin/python3
# -*- coding: utf-8 -*-
from model import Model
from focal_loss import focal_loss_softmax
import tensorflow as tf
import math
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
        embedding = self.embedding_lookup(self.input_ids, self.config['id_word'], embedding_size=self.config['word_dim'], initializer_range=0.02,
                         word_embedding_name="embedding_table", emb_path=self.config['emb_file'])
        self.variable_summaries('embedding', embedding)
        embedded_words_expanded = self. expand_dims(embedding, -1)
        logits, predict_label_ids, l2_loss,author_loss = self.build_cnn(embedded_words_expanded)
        return logits, predict_label_ids, l2_loss,author_loss
    #build_loss(self, labels, logits, l2_loss=0.0)
    def build_loss(self, labels, logits, l2_loss=0,author_loss=0):
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
        return 0.3*loss+0.7*author_loss

    def build_cnn(self, input_tensor):
        pooled_outputs = []
        l2_loss = tf.constant(0.0)  # 先不用，写0
        for filter_size in self.config['filter_sizes']:
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # 卷积层
                filter_shape = [filter_size, self.config['word_dim'], 1, self.config['num_filters']]
                w =  self.initialize_weight("w", filter_shape)
                conv = tf.nn.conv2d(input_tensor, w, strides=[1, 1, 1, 1],
                                    padding="VALID", name="conv")  # 未使用全零填充
                # 加BN层
                conv =  self.batch_norm(conv)
                # intermediate_act_fn = self.get_activation('relu')
                # relu = tf.layers.dense(
                #     conv,
                #     self.config['num_filters'],
                #     activation=intermediate_act_fn,
                #     kernel_initializer=tf.zeros_initializer())
                b =  self.initialize_bias("b", self.config['num_filters'])
                relu = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 池化
                pooled = tf.nn.max_pool(relu, ksize=[1, self.config['max_length'] - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding="VALID", name="pool")
                pooled_outputs.append(pooled)
        num_filters_total = self.config['num_filters'] * len(self.config['filter_sizes'])
        h_pool = tf.concat(pooled_outputs, -1)  # 按照第四维进行连接，h_pool的shape为[batch_size,1,1,num_filters_total]
        output_layer  = tf.reshape(h_pool, [-1, num_filters_total])  # 扁平化数据，跟全连接层相连


        output_layer = gelu(output_layer)
        output_layer = self.dropout(output_layer, self.config['dropout_prob'])
        hidden_size = output_layer.shape[-1].value
        print("======================================================{}============{}====================={}".format(pooled_outputs,self.config['filter_sizes'],num_filters_total,output_layer.shape))
        with tf.variable_scope("output"):
            output_w = tf.get_variable("output_w", shape=[hidden_size, self.config['label_size']])
            output_b =  self.initialize_bias("output_b", shape=self.config['label_size'])
            logits = tf.nn.xw_plus_b(output_layer, output_w, output_b)
            l2_loss += tf.nn.l2_loss(output_w) + tf.nn.l2_loss(output_b)

        with tf.variable_scope("author"):
            # output_w = tf.get_variable("output_w", shape=[hidden_size, self.config['label_size']])
            # output_b =  self.initialize_bias("output_b", shape=self.config['label_size'])
            # author_logits = tf.nn.xw_plus_b(output_layer, output_w, output_b)1

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([self.config['author_size'], hidden_size],
                                    stddev=1.0 / math.sqrt(hidden_size)))
            nce_biases = tf.Variable(tf.zeros([self.config['author_size']]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.

            author_loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=self.author_id,
                               inputs=output_layer,
                               num_sampled=1000,
                               num_classes=self.config['author_size']))


        predict_label_ids = tf.argmax(logits, axis=1, name="predict_label_id")  # 预测结果
        return logits, predict_label_ids, l2_loss,author_loss

