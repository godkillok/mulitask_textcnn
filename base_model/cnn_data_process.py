#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from data_process import DataProcessor

class CnnDataProcessor(DataProcessor):
    def __init__(self):
        super(CnnDataProcessor, self).__init__()

    def get_examples(self, input_dir, mode):
        all_sentences_list = []
        all_labels_list = []
        list_files = os.listdir(input_dir)
        for file in list_files:
            abs_path = os.path.join(input_dir, file)
            sentences_list, labels_list = self._read_json_file(abs_path)
            all_sentences_list.extend(sentences_list)
            all_labels_list.extend(labels_list)
        return all_sentences_list, all_labels_list  # 返回一个二维列表，每个元素是一个词

    def count_word_num(self, sentence_list):
        """
        Count the number of each word in sentences.
        Args:
            sentence_list:
        """
        word_count = {}
        author_count = {}
        category_count = {}
        keyword_count = {}
        max_length = 0
        for sentence in sentence_list:
            temp_len = len(sentence[0])
            max_length = max(temp_len, max_length)

            for i in range(temp_len):
                if sentence[0][i] not in word_count:
                    word_count[sentence[0][i]] = 1
                else:
                    word_count[sentence[0][i]] += 1
            if sentence[1] not in author_count:
                author_count[sentence[1]] = 1
            else:
                author_count[sentence[1]] += 1

            for category in sentence[2]:
                if category not in category_count:
                    category_count[category] = 1
                else:
                    category_count[category] += 1

            for keyword in sentence[3]:
                if keyword not in keyword_count:
                    keyword_count[keyword] = 1
                else:
                    keyword_count[keyword] += 1

        sorted_words = sorted(word_count.items(), key=lambda x: (-x[1], x[0]))  # 按词出现次数从小到大排序
        sorted_authors = sorted(author_count.items(), key=lambda x: (-x[1], x[0]))  # 按author出现次数从小到大排序
        sorted_categorys = sorted(category_count.items(), key=lambda x: (-x[1], x[0]))  # 按category出现次数从小到大排序
        sorted_keywords = sorted(keyword_count.items(), key=lambda x: (-x[1], x[0]))  # 按keyword出现次数从小到大排序
        return sorted_words, sorted_authors, sorted_categorys, sorted_keywords, max_length

    def count_average_length(self, sentence_list):
        """Count the average length of all sentences.
        Args:
            sentence_list: List of sentences consisting of words.
        """
        sum_len = 0
        for sentence in sentence_list:
            sum_len += len(sentence[0])
        return sum_len // len(sentence_list)

    def create_word_dict(self, word_items, min_count_word=2):
        """
        Create a dictionary of word_list from a list of list of word_list.
        Args:
            word_items:
            min_count_word:
        """
        # Only keep most frequent tokens
        word_count = 0
        for word, count in word_items:
            if count > min_count_word:
                self.word_id_dict[word] = word_count
                word_count += 1

    def update_word_dict(self, word_count):
        """
        Update the dictionary with new sentence list.
         Args:
            vocab:
            self.word_id_dict: test/valid word_count
        """
        length = len(self.word_id_dict)
        for word, count in word_count:
            if word not in self.word_id_dict:
                self.word_id_dict[word] = length
                length += 1

    def create_author_dict(self, author_items, min_count_word=3):
        """
        Create a dictionary of word_list from a list of list of word_list.
        Args:
            word_items:
            min_count_word:
        """
        # Only keep most frequent tokens
        author_count = 0
        for word, count in author_items:
            if count > min_count_word:
                self.author_id_dict[word] = author_count
                author_count += 1

    def update_author_dict(self, author_count):
        """
        Update the dictionary with new sentence list.
         Args:
            vocab:
            self.word_id_dict: test/valid word_count
        """
        length = len(self.author_id_dict)
        for author, count in author_count:
            if author not in self.author_id_dict:
                self.author_id_dict[author] = length
                length += 1

    def append_author_unk_dict(self):
        """Add the unk and pad numbers to the dictionary."""
        author_count = len(self.author_id_dict)
        # Add unk tokens.
        if self.unk not in self.author_id_dict:
            self.author_id_dict[self.unk] = author_count

    def create_category_dict(self, category_items, min_count_word=0):
        """
        Create a dictionary of word_list from a list of list of word_list.
        Args:
            word_items:
            min_count_word:
        """
        # Only keep most frequent tokens
        category_count = 0
        for category, count in category_items:
            if count > min_count_word:
                self.category_id_dict[category] = category_count
                category_count += 1

    def update_category_dict(self, category_count):
        """
        Update the dictionary with new sentence list.
         Args:
            vocab:
            self.word_id_dict: test/valid word_count
        """
        length = len(self.category_id_dict)
        for category, count in category_count:
            if category not in self.category_id_dict:
                self.category_id_dict[category] = length
                length += 1

    def append_category_unk_dict(self):
        """Add the unk and pad numbers to the dictionary."""
        category_count = len(self.category_id_dict)
        # Add unk tokens.
        if self.unk not in self.category_id_dict:
            self.category_id_dict[self.unk] = category_count

    def create_keyword_dict(self, keyword_items, min_count_word=0):
        """
        Create a dictionary of word_list from a list of list of word_list.
        Args:
            word_items:
            min_count_word:
        """
        # Only keep most frequent tokens
        keyword_count = 0
        for keyword, count in keyword_items:
            if count > min_count_word:
                self.keyword_id_dict[keyword] = keyword_count
                keyword_count += 1

    def update_keyword_dict(self, keyword_count):
        """
        Update the dictionary with new sentence list.
         Args:
            vocab:
            self.word_id_dict: test/valid word_count
        """
        length = len(self.keyword_id_dict)
        for keyword, count in keyword_count:
            if keyword not in self.keyword_id_dict:
                self.keyword_id_dict[keyword] = length
                length += 1

    def append_keyword_pad_dict(self):
        """Add the pad numbers to the dictionary."""
        keyword_count = len(self.keyword_id_dict)
        self.keyword_id_dict[self.pad] = keyword_count

    def get_words(self):
        """Gets the list of tags for this data set."""
        for key in self.word_id_dict:
            self.word_list.append(key)
        return self.word_list
    
    def get_labels(self):
        label_list = []
        """Gets the list of tags for this data set."""
        for key in self.label_id_dict:
            label_list.append(key)
        return label_list

    def get_id_words(self):
        """Gets the list of tags for this data set."""
        for key in self.word_id_dict:
            self.id_word_dict[self.word_id_dict[key]] = key
        return self.id_word_dict


    def convert2tfrecord(self, sentence_list, label_list, target_dir, max_length, max_keyword_length):
        """Saves dict to json file
        Args:
            target_dir: target folder
            max_length: max length
            name_dir: folder name to write
        """
        assert len(sentence_list) == len(label_list)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        # 创建一个writer来写TFRecord文件
        block_count = 5000
        count = 0
        # tfrecords格式文件名
        filename = os.path.join(target_dir, str(count) + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename)
        for sentence, label in zip(sentence_list, label_list):
            word_list = sentence[0]
            author = sentence[1]
            category_list = sentence[2]
            keyword_list = sentence[3]
            keyword_label_list = sentence[4]
            count += 1
            word_id_list = []
            keyword_id_list = []
            keyword_label_id_list = []
            category_id_list = []
            true_len = len(word_list)
            for word in word_list:
                if word in self.word_id_dict:
                    word_id_list.append(self.word_id_dict[word])
                else:
                    word_id_list.append(self.word_id_dict[self.unk])
            if true_len > max_length:
                word_id_list = word_id_list[:max_length]
            else:
                word_id_list += [self.word_id_dict[self.pad]] * (max_length - true_len)
            assert len(word_id_list) == max_length

            if author in self.author_id_dict:
                author_id = self.author_id_dict[author]
            else:
                author_id = self.author_id_dict[self.unk]
            
            for category in category_list:
                if category in self.category_id_dict:
                    category_id_list.append(self.category_id_dict[category])
                else:
                    category_id_list.append(self.category_id_dict['<unk>'])
            
            if len(keyword_list) > max_keyword_length:
                keyword_list = keyword_list[0:max_keyword_length]

            for keyword in keyword_list:
                keyword_id_list.append(self.keyword_id_dict[keyword])

            while len(keyword_id_list) < max_keyword_length:
                keyword_id_list.append(self.keyword_id_dict['<pad>'])

            assert len(keyword_id_list) == max_keyword_length

            if len(keyword_label_list) > max_keyword_length:
                keyword_label_list = keyword_label_list[0:max_keyword_length]

            for keyword_label in keyword_label_list:
                keyword_label = keyword_label.split('#')[1]
                if keyword_label == 'entertainment&industry':
                    keyword_label = 'entertainment_industry'
                if keyword_label == 'music':
                    keyword_label = 'music&audio'
                if keyword_label in self.label_id_dict:
                    keyword_label_id_list.append(self.label_id_dict[keyword_label])

            while len(keyword_label_id_list) < max_keyword_length:
                keyword_label_id_list.append(self.label_id_dict['<pad>'])

            assert len(keyword_label_id_list) == max_keyword_length

            label_id = self.label_id_dict[label]
            example = tf.train.Example(features=tf.train.Features(feature={
                'sentence_ids': self._int64_feature(word_id_list),
                'author_id': self._int64_feature(author_id),
                'category_ids': self._int64_feature(category_id_list),
                'keyword_ids': self._int64_feature(keyword_id_list),
                'keyword_label_ids': self._int64_feature(keyword_label_id_list),
                'label_id': self._int64_feature(label_id)}))
            # 将example写入TFRecord文件
            if count % block_count == 0:
                filename = os.path.join(target_dir, str(count) + '.tfrecords')
                writer = tf.python_io.TFRecordWriter(filename)
            writer.write(example.SerializeToString())
        writer.close()
