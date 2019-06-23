import re
import json
punctuation = r"""!"#$%&()*+,’”-./:;<=>?@[\]^_`{|}~。，"""

def clean_str(text):
    text = text.strip()
    text = re.sub(r'[{}]+'.format(punctuation), ' ', str(text))
    text=' '.join(text.split())
    return text

def per_line(line):
    li = json.loads(line)
    pic_tags = [pic.split('\x02')[0] for pic in li.get('pic_tags', '').split('\x03')]
    fields = li.get("title", "") + ' ' + ' '.join(li.get("tags", [""])) + ' '.join(pic_tags)
    text = clean_str(fields.lower())
    text = text.replace("\n", ' ').replace("\r", " ")
    return text