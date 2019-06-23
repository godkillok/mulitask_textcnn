from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import re
import os
import json
# import logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# from cnn import nn_pred
# for python 22.x1
# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
#model_dir2 is the good one embedding size 1281
flags = tf.app.flags
flags.DEFINE_string("model_dir", "/data/tanggp/text_cnns/data/model1/", "Base directory for the model.")
flags.DEFINE_string("train_file_pattern", "/data/tanggp/text_cnns/data/*train.tfrecords", "train file pattern")
flags.DEFINE_string("eval_file_pattern", "/data/tanggp/text_cnns/data/*test.tfrecords", "evalue file pattern")
# flags.DEFINE_string("train_file_pattern", "s3://shareit.tmp.ap-southeast-1/NeverDelete/tanggp/tfrecord/train*.tfrecords", "train file pattern")
# flags.DEFINE_string("eval_file_pattern", "s3://shareit.tmp.ap-southeast-1/NeverDelete/tanggp/tfrecord/train*.tfrecords", "evalue file pattern")

flags.DEFINE_string("pred_eval_file_pattern", "/data/tanggp/text_cnns/data/*test.tfrecords", "evalue file pattern")


flags.DEFINE_float("dropout_rate", 0.5, "Drop out rate")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate")
flags.DEFINE_float("decay_rate", 0.8, "L+earning rate")
flags.DEFINE_integer("embedding_size", 100, "embedding size")
flags.DEFINE_integer("num_filters", 200, "number of filters")
flags.DEFINE_integer("num_classes", 20, "number of classes")
flags.DEFINE_integer("num_categories", 20, "number of classes")
flags.DEFINE_integer("num_parallel_readers", 8, "number of classes")
flags.DEFINE_integer("shuffle_buffer_size", 300, "dataset shuffle buffer size")
flags.DEFINE_integer("sentence_max_len", 200, "max length of sentences")
flags.DEFINE_integer("batch_size", 64, "number of instances in a batch")
flags.DEFINE_integer("save_checkpoints_steps", 3000, "Save checkpoints every this many steps")
flags.DEFINE_integer("train_steps", 100000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("decay_steps", 5000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("train_epoch", 10000,
                     "Number of (global) training steps to perform")
flags.DEFINE_string("data_dir", "/data/tanggp/youtube8m/",
                    "Directory containing the dataset")
flags.DEFINE_string("test_dir", " /data/tanggp/youtube8m/",
                    "Directory containing the dataset")
flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated list of number of window size in each filter")
flags.DEFINE_string("pad_word", "<pad>", "used for pad sentence")
flags.DEFINE_string("path_vocab", "/data/tanggp/text_cnns/data/words.txt", "used for word index")
flags.DEFINE_string("fast_text", "/data/tanggp/text_cnns/data/super_more.bin", "used for word index")
flags.DEFINE_string("work_type", "", "used for word index")
flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')

FLAGS = flags.FLAGS
BUCKET_NAME="recommendation.ap-southeast-1"

def parse_exmp(serialized_example):
    feats = tf.parse_single_example(
        serialized_example,
        features={
            "text": tf.FixedLenFeature([FLAGS.sentence_max_len], tf.int64),
            # {"text": text, "label": label, "author": author, "categories": categories}
            "categories": tf.FixedLenFeature([], tf.int64),
            "author": tf.FixedLenFeature([], tf.int64),

            "label": tf.FixedLenFeature([], tf.int64)
        })

    labels = feats.pop('label')

    return feats, labels


def train_input_fn(filenames, shuffle_buffer_size,shuffle=True,repeat=0):

    # Load txt file, one example per line
    files = tf.data.Dataset.list_files(filenames)  # A dataset of all files matching a pattern.
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size).repeat(FLAGS.train_epoch)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_exmp, batch_size=FLAGS.batch_size,
                                                          num_parallel_calls=2))
    return dataset

def assign_pretrained_word_embedding(params):
    print("using pre-trained word emebedding.started.word2vec_model_path:",FLAGS.fast_text)
    # import fastText as ft
    # word2vec_model = ft.load_model(FLAGS.fast_text)

    vocab_size=params["vocab_size"]

    word_embedding_final = np.zeros((vocab_size,FLAGS.embedding_size))  # create an empty word_embedding list.

    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0
    count_not_exist = 0

    with open(FLAGS.path_vocab, 'r', encoding='utf8') as f:
        lines = f.readlines()
        vocab = {l.strip(): i for i, l in enumerate(lines)}
    print(len(vocab))
    print(len(lines))
    print(vocab_size)
    #assert len(vocab)==vocab_size

    for (word, idx) in vocab.items():
        # embedding=word2vec_model.get_word_vector(word)
        # if embedding is not None:  # the 'word' exist a embedding
        #     word_embedding_final[idx,:] = embedding
        #     count_exist = count_exist + 1  # assign array to this word.
        # else:  # no embedding for this word
        word_embedding_final[idx, :] = np.random.uniform(-bound, bound, FLAGS.embedding_size);
        count_not_exist = count_not_exist + 1  # init a random value for the word.

    # word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor

    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")
    return word_embedding_final



def my_model(features, labels, mode, params):

    sentence = features['text']

    categories=features["categories"]
    author = features["author"]

    # Get word embeddings for each token in the sentence

    assert (params["vocab_size"], FLAGS.embedding_size) == params["word_embedding"].shape
    embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                                 shape=[params["vocab_size"], FLAGS.embedding_size],
                                 initializer=tf.constant_initializer(params["word_embedding"], dtype=tf.float32))

    sentence = tf.nn.embedding_lookup(embeddings, sentence)  # shape:(batch, sentence_len, embedding_size)
    # sentence_contact = [sentence,segment,tfidf]
    # sentence=tf.concat(sentence_contact, 2)  # shape: (batch, 1, len(filter_size) * embedding_size, 1)
    # print("this---------shape---{}".format(sentence.shape))
    filter_len=sentence.shape[2]
    # time.sleep(10)
    # add a channel dim, required by the conv2d and max_pooling2d method
    sentence = tf.expand_dims(sentence, -1)  # shape:(batch, sentence_len/height, embedding_size/width, channels=1)
    # sentence = tf.layers.batch_normalization(sentence, training=(mode == tf.estimator.ModeKeys.TRAIN))
    pooled_outputs = []
    for filter_size in params["filter_sizes"]:

        conv = tf.layers.conv2d(
            sentence,
            filters=FLAGS.num_filters,
            kernel_size=[filter_size, filter_len],
            strides=(1, 1),
            padding="VALID"
        )  # activation=tf.nn.relu
        # conv = tf.layers.batch_normalization(conv, training=(mode == tf.estimator.ModeKeys.TRAIN))
        conv = tf.nn.relu(conv)

        # conv = tf.layers.conv2d(
        #     conv,
        #     filters=FLAGS.num_filters,
        #     kernel_size=[filter_size, 1],
        #     strides=(1, 1),
        #     padding="SAME"
        # )#activation=tf.nn.relu
        # conv = tf.nn.relu(conv)
        # b = tf.get_variable("b-%s" % filter_size, [FLAGS.num_filters])  # ADD 2017-06-09
        if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
            # h_pool_flat = tf.layers.batch_normalization(h_pool_flat, training=(mode == tf.estimator.ModeKeys.TRAIN))

            conv = tf.layers.dropout(conv, params['dropout_rate'],
                                     training=(mode == tf.estimator.ModeKeys.TRAIN))
        # conv = tf.layers.batch_normalization(conv, training=(mode == tf.estimator.ModeKeys.TRAIN))

        pool = tf.layers.max_pooling2d(
            conv,
            pool_size=[FLAGS.sentence_max_len - filter_size + 1, 1],
            strides=(1, 1),
            padding="VALID")

        pooled_outputs.append(pool)

    h_pool = tf.concat(pooled_outputs, 3)  # shape: (batch, 1, len(filter_size) * embedding_size, 1)
    h_pool_flat = tf.reshape(h_pool, [-1, FLAGS.num_filters * len(
        params["filter_sizes"])])  # shape: (batch, len(filter_size) * embedding_size)
    if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
        # h_pool_flat = tf.layers.batch_normalization(h_pool_flat, training=(mode == tf.estimator.ModeKeys.TRAIN))

        h_pool_flat = tf.layers.dropout(h_pool_flat, params['dropout_rate'],
                                        training=(mode == tf.estimator.ModeKeys.TRAIN))
    h_pool_flat = tf.layers.batch_normalization(h_pool_flat, training=(mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.layers.dense(h_pool_flat, FLAGS.num_classes, activation=None)

    # learning_rate = tf.train.exponential_decay(params['learning_rate'], tf.train.get_global_step(), FLAGS.decay_steps,
    #                                            FLAGS.decay_rate, staircase=True)
    learning_rate = params['learning_rate']
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

    def _train_op_fn(loss):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # def loss_fn_(labels, logits):  # 0.001
    #     l2_lambda = 0.0001
    #     with tf.name_scope("loss"):
    #         # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
    #         # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
    #         losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
    #                                                                 logits=logits);  # sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
    #         # print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
    #         loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
    #         l2_losses = tf.add_n(
    #             [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
    #         loss = loss + l2_losses
    #     return loss

    my_head = tf.contrib.estimator.multi_class_head(n_classes=FLAGS.num_classes)
    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        logits=logits,
        train_op_fn=_train_op_fn
    )


def main(unused_argv):
    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(FLAGS.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
    # Loads parameters from json file
    with open(json_path) as f:
        config = json.load(f)
    FLAGS.pad_word = config["pad_word"]
    if config["train_size"] < FLAGS.shuffle_buffer_size:
        FLAGS.shuffle_buffer_size = config["train_size"]
    print("shuffle_buffer_size:", FLAGS.shuffle_buffer_size)

    # Get paths for vocabularies and dataset

    # path_train = os.path.join(FLAGS.data_dir, 'train.csv')
    # path_eval = os.path.join(FLAGS.data_dir, 'test.csv')
    path_train = FLAGS.train_file_pattern
    path_eval = FLAGS.eval_file_pattern
    params={
        'vocab_size': config["vocab_size"],
        'filter_sizes': list(map(int, FLAGS.filter_sizes.split(','))),
        'learning_rate': FLAGS.learning_rate,
        'dropout_rate': FLAGS.dropout_rate
    }
    word_embedding = assign_pretrained_word_embedding(params)
    params['word_embedding']=word_embedding
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params=params,
        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)

    )

    # train_spec = tf.estimator.TrainSpec(
    #     input_fn=lambda: input_fn(path_train, path_words, FLAGS.shuffle_buffer_size, config["num_oov_buckets"]),
    #     max_steps=FLAGS.train_steps
    # )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(path_train, FLAGS.shuffle_buffer_size, repeat=FLAGS.train_epoch),
        max_steps=FLAGS.train_steps
    )

    # input_fn_for_eval = lambda: input_fn(path_eval, path_words, 0, config["num_oov_buckets"])
    input_fn_for_eval = lambda: train_input_fn(path_eval, 0,shuffle=False)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, steps=100, throttle_secs=100)

    print("before train and evaluate")
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


    # print("evalue train set")
    # input_fn_for_pred = lambda: train_input_fn(path_train, 0)
    # classifier.evaluate(input_fn=input_fn_for_pred,steps=120)
    #
    # input_fn_for_eval = lambda: train_input_fn(path_eval, 0)
    # # eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, steps=30, throttle_secs=60)
    # # tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    #
    # print("evalue eval set")
    # classifier.evaluate(input_fn=input_fn_for_eval,steps=100)
    # print("after train and evaluate")

if __name__ == "__main__":
    import time
    # logging.info('begin train')
    t1=time.time()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
    # logging.info('end train')
    t2 = time.time()
    print('end train {}'.format(t2-t1))
    # tf.app.run(main=nn_pred.pred(my_model,FLAGS,'cnn2'))
