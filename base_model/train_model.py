#!/usr/bin/python3
# -*- coding: utf-8 -*-


import tensorflow as tf
from cnn_model import CnnModel
import optimization
from tf_metrics import precision, recall, f1

def model_fn(features, labels, mode, params):
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    input_ids = features["text"]
    author_id = features["author"]
    category_ids = features["categories"]
    label_id = features["label"]
    cnn = CnnModel(params, input_ids, author_id, category_ids, training)
    logits, predict_label_ids, l2_loss = cnn.build_network()
    squeeze_label_ids = tf.squeeze(label_id, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        #words = tf.contrib.lookup.index_to_string_table_from_file(params['vocab'])
        #input_words = words.lookup(tf.to_int64(input_ids))
        predictions = {
            'true_label_ids': squeeze_label_ids,
            'predict_label_ids': predict_label_ids,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        loss = tf.cast(cnn.build_loss(squeeze_label_ids, logits, l2_loss), dtype=tf.float32)
        train_op = optimization.create_optimizer(loss, params['learning_rate'], params['train_steps'], params['num_warmup_steps'])
        if mode == tf.estimator.ModeKeys.EVAL:
            # Metrics
            metrics = {
                'acc': tf.metrics.accuracy(squeeze_label_ids, predict_label_ids),
                # 分别计算各个类的P, R 然后按类求平均值
                'precision': precision(squeeze_label_ids, predict_label_ids, params['label_size'], average='macro'),
                'recall': recall(squeeze_label_ids, predict_label_ids, params['label_size'], average='macro'),
                'f1': f1(squeeze_label_ids, predict_label_ids, params['label_size'], average='macro'),
            }
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
