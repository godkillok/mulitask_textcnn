#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json
import tensorflow as tf

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self):
        self.unk = '<unk>'
        self.pad = '<pad>'
        self.label_id_dict = {}
        self.word_list = []
        self.word_id_dict = {}
        self.id_word_dict = {}
        self.author_id_dict = {}
        self.category_id_dict = {}
        self.keyword_id_dict = {}

    @classmethod
    def _read_json_file(cls, input_file):
        """Read a json file.
        Args:
            input_file: input file.
        """
        sentences_list = []
        label_list = []
        with open(input_file, "r", encoding='utf8') as f:
            for line in f:
                line = line.strip()
                content = json.loads(line)
                text = ''
                author = ''
                google_label = ''
                category = ''
                label = ''
                keyword_list = []
                keyword_label_list = []
                if content.get('title'):
                    text = content['title']

                if content.get('tags'):
                    text += ' ' + ' '.join(content['tags'])

                if content.get('content'):
                    text = content['content']

                if content.get('label'):
                    label = content['label'].split('#')[1]

                if content.get('google_label'):
                    google_label = content['google_label'][0]

                if content.get('categories'):
                    category = content['categories'][0]

                if content.get('source_user'):
                    author = content['source_user']

                if content.get('keyword'):
                    keyword_list = content['keyword']

                if content.get('keyword_label'):
                    keyword_label_list = content['keyword_label']
                
                text_list = text.split()
                sentences_list.append((text_list, author,[google_label, category], keyword_list, keyword_label_list))
                label_list.append(label)
        return sentences_list, label_list  # 返回一个二维列表，每个元素是一个词

    @classmethod
    def _read_file(cls, input_file, delimiter='\t'):
        """Read a tab separated value file.
        Args:
            input_file: input file.
            delimiter:
        """
        sentences_list = []
        label_list = []
        with open(input_file, "r", encoding='utf8') as f:
            for line in f:
                line = line.strip()
                content, label = line.split(delimiter)
                line_list = content.split(' ')
                sentences_list.append(line_list)
                label_list.append(label)
        return sentences_list, label_list  # 返回一个二维列表，每个元素是一个词

    def get_examples(self, input_dir, mode):
        """Gets a collection of `InputExample`s for the given folder.
        Args:
            input_dir: input folder.
            mode: train/test/valid.
        """
        raise NotImplementedError()

    def count_word_num(self, sentence_list):
        """Count the number of each word in sentences.
        Args:
            sentence_list: List of sentences consisting of words.
        """
        raise NotImplementedError()

    def count_average_length(self, sentence_list):
        """Count the average length of all sentences.
        Args:
            sentence_list: List of sentences consisting of words.
        """
        raise NotImplementedError()

    def create_word_dict(self, word_items, min_count_word=2):
        """Create a dictionary based on the number of occurrences of each word.
        Args:
            word_items: Combination of words and their occurrences.
            min_count_word: The minimum number of occurrences of a word
        """
        raise NotImplementedError()

    def update_word_dict(self, word_count):
        """Update the dictionary with new sentence list.
         Args:
            word_count: The number of occurrences of a word.
        """
        raise NotImplementedError()

    def append_unk_pad_dict(self):
        """Add the unk and pad numbers to the dictionary."""

        word_count = len(self.word_id_dict)
        # Add unk tokens.
        if self.unk not in self.word_id_dict:
            self.word_id_dict[self.unk] = word_count
            word_count += 1

        # Add pad tokens.
        if self.pad not in self.word_id_dict:
            self.word_id_dict[self.pad] = word_count

    def create_label_dict(self, labels):
        """
        Create a dictionary of label_id_dict based on a list of labels.
        Args:
            labels: label list.
        """
        labels = set(labels)
        for index, label in enumerate(labels):
            self.label_id_dict[label] = index

    def append_label_pad_dict(self):
        """Add the pad numbers to the dictionary."""
        label_count = len(self.label_id_dict)
        self.label_id_dict[self.pad] = label_count

    def get_labels(self):
        """Get the list of labels for this data set."""
        raise NotImplementedError()

    def get_words(self):
        """Get the list of words for this data set."""
        raise NotImplementedError()

    def get_id_words(self):
        """Get the dictionary of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _int64_feature(cls, value):
        if isinstance(value, int):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        elif isinstance(value, list):
            try:
                tf.train.Feature(int64_list=tf.train.Int64List(value=value))
            except:
                print(value)
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @classmethod
    def _bytes_feature(cls, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @classmethod
    def _float_feature(cls, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def convert2tfrecord(self, sentence_list, label_list, target_dir, max_length, max_keyword_length):
        """Convert txt file to tfrecord file.
        Args:
            input_dir: input folder.
            target_dir: folder name to write.
            max_length: max length.
        """
        raise NotImplementedError()

    def save_vocab_to_txt_file(self, vocab, txt_path):
        """Writes one token per line, 0-based line word_id corresponds to the word_id of the token.
        Args:
            vocab: (iterable object) yields token.
            txt_path: (string) path to vocab file.
        """
        with open(txt_path, "w", encoding='utf8') as f:
            f.write("\n".join(token for token in vocab))

    def save_dict_to_json(self, d, json_path):
        """Save dict to json file.
        Args:
            d: (dict).
            json_path: path to json file.
        """
        with open(json_path, 'w', encoding='utf8') as f:
            d = {k: v for k, v in d.items()}
            json.dump(d, f, indent=4)
