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
    with open(txt_path, "w") as f:
        f.write("\n".join(token for token in vocab))

def save_author_to_txt_file(author, txt_path):
    """
    Writes one token per line, 0-based line id corresponds to the id of the token.
    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, "w") as f:
        for vo in author:
            for k,v in vo.items():
                f.write("{}\x01\t{}\n".format(k,v))

def save_dict_to_json(d, json_path):
    """Saves dict to json file
    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_author(txt_path, author):
    """Update word and tag vocabulary from dataset
    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method
    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path) as f:
        for i, line in enumerate(f):
            li=json.loads(line)
            author.append(li.get("source_user"))



if __name__ == '__main__':
    args = parser.parse_args()

    # Build word vocab with train and test datasets
    print("Building author...")
    author = []
    update_author(os.path.join(args.data_dir, 'txt_train'), author)
    update_author(os.path.join(args.data_dir, 'txt_valid'), author)

    author_sort=sorted(Counter(author).items(), key=lambda x: x[1], reverse=True)
    print('author num {}'.format(len(author_sort)))
    save_author_to_txt_file(author_sort, os.path.join(args.data_dir, 'textcnn_author_sort'))
    print("- done.")
    os.system("head {}".format( os.path.join(args.data_dir, 'textcnn_author_sort')))