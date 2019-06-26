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
        embedding = self.embedding_lookup(self.input_ids, self.config['id_word'], embedding_size=self.config['word_dim'], initializer_range=0.02,
                         word_embedding_name="embedding_table", emb_path=self.config['emb_file'])
        self.variable_summaries('embedding', embedding)
        embedded_words_expanded = self. expand_dims(embedding, -1)
        logits, predict_label_ids, l2_loss = self.build_cnn(embedded_words_expanded)
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

        if self.config['use_author_feature']:
            author_embedding_table = tf.get_variable(name='author_embedding_name',
                                                     shape=[self.config['author_size'], self.config['feature_dim']],
                                                     initializer=tf.keras.initializers.he_normal(), dtype=tf.float32)
            author_embedding = tf.reshape(tf.nn.embedding_lookup(author_embedding_table, self.author_id),
                                          [-1, self.config['feature_dim']])

            output_layer = tf.concat([output_layer , author_embedding], axis=1)

        if self.config['use_category_feature']:
            category_embedding_table = tf.get_variable(name='category_embedding_name',
                                                       shape=[self.config['category_size'], self.config['feature_dim']],
                                                       initializer=tf.keras.initializers.he_normal(), dtype=tf.float32)

            category_embedding = tf.reshape(tf.nn.embedding_lookup(category_embedding_table, self.category_ids),
                                            [-1, 2 * self.config['feature_dim']])

            output_layer = tf.concat([output_layer, category_embedding], axis=1)

        if self.config['use_keyword_feature']:
            keyword_embedding_table = tf.get_variable(name='keyword_embedding_name',
                                                      shape=[self.config['keyword_size'], self.config['feature_dim']],
                                                      initializer=tf.keras.initializers.he_normal(), dtype=tf.float32)
            keyword_label_embedding_table = tf.get_variable(name='label_embedding_name',
                                                            shape=[self.config['label_size'] + 1, self.config['feature_dim']],
                                                            initializer=tf.keras.initializers.he_normal(),
                                                            dtype=tf.float32)
            keyword_embedding = tf.nn.embedding_lookup(keyword_embedding_table, self.keyword_ids)

            keyword_label_embedding = tf.nn.embedding_lookup(keyword_label_embedding_table, self.keyword_label_ids)

            sum_keyword_embedding = tf.reduce_mean(keyword_embedding, axis=1)
            sum_keyword_label_embedding = tf.reduce_mean(keyword_label_embedding, axis=1)

            output_layer = tf.concat([output_layer, sum_keyword_embedding, sum_keyword_label_embedding], axis=1)
        output_layer = gelu(output_layer)
        output_layer = self.dropout(output_layer, self.config['dropout_prob'])
        hidden_size = output_layer.shape[-1].value
        with tf.variable_scope("output"):
            output_w = tf.get_variable("output_w", shape=[hidden_size, self.config['label_size']])
            output_b =  self.initialize_bias("output_b", shape=self.config['label_size'])
            logits = tf.nn.xw_plus_b(output_layer, output_w, output_b)
            l2_loss += tf.nn.l2_loss(output_w) + tf.nn.l2_loss(output_b)

        predict_label_ids = tf.argmax(logits, axis=1, name="predict_label_id")  # 预测结果
        return logits, predict_label_ids, l2_loss

