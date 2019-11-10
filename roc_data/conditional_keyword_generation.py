# -*- coding:UTF-8 -*-
#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
from collections import Counter
import numpy as np
import nltk
import json

def process_data(all_story, write_fname):
	f = open (write_fname, 'w')
	for story in all_story:
		#key_array = story['title'] + ["<EOT>"] + story['keyword'] + ["<EOK>"]
		key_dict = {}
		key_dict["x_tokens"] = story['title']
		key_dict["y_tokens"] = story['title'] + ["<EOT>"] + story['keyword'] + ["<EOK>"]
		f.write(str(json.dumps(key_dict))+'\n')
	f.close()

def get_all_words(all_story):
	all_words = []
	for story in all_story:
		all_words.extend(story['title'])
		all_words.extend(story['keyword'])
	return all_words

def build_vocab(all_words, max_vocab_cnt = 30000):
	#vocab/itos: all the word in array
    #rev_vocab/stoi: words with id (type = dict)
    #rev_vocab/tok2i: {word: id}
    #vocab_dict/i2tok: {id: word}
    vocab_count = Counter(all_words).most_common()
    print(vocab_count[:10])
    raw_vocab_size = len(vocab_count)
    print(raw_vocab_size)
    #discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
    #print(discard_wc)
    vocab_count = vocab_count[0:max_vocab_cnt]
    print(vocab_count[:10])
    print(len(vocab_count))
    vocab = ["<unk>", "<pad>", "<init>", "<eos>", ","] + [t for t, cnt in vocab_count]
    rev_vocab = {t: idx for idx, t in enumerate(vocab)}
    vocab_dict = {idx: t for idx, t in enumerate(vocab)}

    #print(vocab[:10])
    #print(rev_vocab)
    #print(vocab_dict)

def get_roc_graph(path):
	all_story = []
	with open (path, 'r') as file:
		stories = file.readlines()
		for story in stories:
			story_dict = {}
			story_dict['title'] = story.split("<EOT>")[0].strip().split()
			story_dict['keyword'] = story.split("<EOT>")[1].split("<EOL>")[0].strip().split()
			#story_arr = story.split("<EOT>")[1].split("<EOL>")[1].strip().split('</s>')
			#story_new_arr = []
			#for line in story_arr:
			#	if line != '':
			#		story_new_arr.append(line.strip().split())
			#story_dict['story'] = story_new_arr
			all_story.append(story_dict)
	return all_story

train_filename = '../rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.train'
valid_filename = '../rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.dev'   
test_filename = '../rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.test'

train_all_story = get_roc_graph(train_filename)
process_data(train_all_story, '../rocstory_plan_write/keyword_data/train.txt')
valid_all_story = get_roc_graph(valid_filename)
process_data(valid_all_story, '../rocstory_plan_write/keyword_data/valid.txt')
test_all_story = get_roc_graph(test_filename)
process_data(test_all_story, '../rocstory_plan_write/keyword_data/test.txt')

all_words = get_all_words(train_all_story + valid_all_story + test_all_story)
print(all_words[:100])
build_vocab(all_words)



