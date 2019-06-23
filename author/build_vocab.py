"""Build vocabularies of words and labels from datasets"""
import argparse
from collections import Counter
import json
import os
import re
from common_tool import per_line
parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=20, help="Minimum count for words in the dataset", type=int)
parser.add_argument('--data_dir', default='/data/tanggp/youtube8m', help="Directory containing the dataset")

# Hyper parameters for the vocab
NUM_OOV_BUCKETS = 1 # number of buckets (= number of ids) for unknown words
PAD_WORD = '0'
label_class=[]


def save_vocab_to_txt_file(vocab, txt_path):
    """
    Writes one token per line, 0-based line id corresponds to the id of the token.
    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(token for token in vocab))


def save_dict_to_json(d, json_path):
    """Saves dict to json file
    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w', encoding="utf8") as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_vocab(txt_path, vocab,word_lenth):
    """Update word and tag vocabulary from dataset
    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method
    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            text=per_line(line)
            tokens = text.split()
            tokens = [w.strip("'") for w in tokens if len(w.strip("'")) > 0 and not w.isdigit()]
            vocab.update(tokens)
            word_lenth.append(len(tokens))
    return i + 1


if __name__ == '__main__':
    args = parser.parse_args()

    # Build word vocab with train and test datasets
    print("Building word vocabulary...")
    words = Counter()
    word_lenth=[]
    size_train_sentences = update_vocab(os.path.join(args.data_dir, 'txt_train'), words,word_lenth)
    size_test_sentences = update_vocab(os.path.join(args.data_dir, 'txt_golden'), words,word_lenth)
    print("- done.")
    word_lenth_count=Counter(word_lenth)
    for i in word_lenth_count.items():
        print(i)

    print('most common 100 {}'.format(words.most_common(100)))
    print('before remove {}'.format(len(words.keys())))
    # Only keep most frequent tokens
    max_word=max(words.values())
    words=[PAD_WORD,'1']
    words += [tok for tok, count in words.items() if count >= args.min_count_word and count<0.95*max_word]
    print('after remove {}'.format(len(words)))
    # Add pad tokens
    #if PAD_WORD not in words: words.append(PAD_WORD)
    # Save vocabularies to file
    print("Saving vocabularies to file...")
    save_vocab_to_txt_file(words, os.path.join(args.data_dir, 'textcnn_words.txt'))
    print("- done.")

    # Save datasets properties in json file
    sizes = {
        'train_size': size_train_sentences,
        'test_size': size_test_sentences,
        'vocab_size': len(words),
        'pad_word': PAD_WORD,
        'num_oov_buckets': NUM_OOV_BUCKETS
    }
    save_dict_to_json(sizes, os.path.join(args.data_dir, 'textcnn_dataset_params.json'))

    # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))