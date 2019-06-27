# -*- coding: utf-8 -*-
import tensorflow as tf

"""
Tensorflow实现何凯明的Focal Loss, 该损失函数主要用于解决分类问题中的类别不平衡
focal_loss_sigmoid: 二分类loss
focal_loss_softmax: 多分类loss
Reference Paper : Focal Loss for Dense Object Detection
"""


def focal_loss_sigmoid(labels, logits, alpha=0.25, gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred = tf.nn.sigmoid(logits)
    labels = tf.to_float(labels)
    loss = -labels * (1 - alpha) * ((1 - y_pred) * gamma) * tf.log(y_pred) - \
        (1 - labels) * alpha * (y_pred ** gamma) * tf.log(1 - y_pred)
    return loss


def focal_loss_softmax(labels, logits, gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred = tf.nn.softmax(logits, dim=-1)  # [batch_size,num_classes]
    labels = tf.one_hot(labels, depth=y_pred.shape[1])
    alpha1=[0.11627907,0.161290323,0.113636364,0.196078431,0.217391304,0.217391304,0.344827586,0.128205128,0.212765957,0.4,0.4,1,0.27027027,0.454545455,0.227272727,0.181818182,0.434782609,0.243902439,0.588235294,0.666666667]
    alpha=tf.constant(value=alpha1, dtype=tf.float32)
    L1=-alpha*labels*((1-y_pred)**gamma)*tf.log(y_pred)
    loss=tf.reduce_sum(L1,axis=1)
    return loss
