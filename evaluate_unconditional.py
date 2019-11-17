import tree_text_gen.binary.unconditional.evaluate as evaluate
from pprint import pprint as pp
import os
import argparse
import json
from glob import glob
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import torch.nn.functional as F
import torch as th
import seaborn as sns
from tree_text_gen.binary.common.data import load_personachat, build_tok2i, SentenceDataset, inds2toks
from torch.utils.data.dataloader import DataLoader
import tree_text_gen.binary.common.samplers as samplers
import tree_text_gen
from tree_text_gen.binary.common.tree import build_tree, tree_to_text, print_tree, Node, Tree

project_dir = os.path.abspath(os.path.join(os.path.dirname(tree_text_gen.__file__), os.pardir))
dirs = glob(os.path.join(project_dir, 'models/unconditional/*'))
d = dirs[0]
CHECKPOINT = True
exprs = {}
for d in dirs:
    expr_name = d.split('/')[-1]
    exprs[expr_name] = d
models = {}
for k, v in exprs.items():
    print(k)
    print(v)
    models[k] = evaluate.load_model(v, k, checkpoint=CHECKPOINT)

def run(topk, generation_count=10, save_dir='./roc_gen_result'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for name, model in models.items():
        print('=== %s ===' % name)
        save_fname = save_dir + '/' + str(name) + '_lm_result_' + 'topk_' + str(topk) + "_gennum_" + str(generation_count) + ".txt"
        print("Save file name: " , save_fname)
        k = topk
        if k == -1:
            model.sampler.eval_sampler = samplers.StochasticSampler()
        else:
            model.sampler.eval_sampler = samplers.TopkSampler(k, model.device)
        out = evaluate.sample(model, generation_count)
        
        with open(save_fname, 'w') as fname:
            for o in out:
                o_dict = {}
                o_dict['sentence'] = ' '.join(o['inorder_tokens'])
                o_dict['generation order'] = ' '.join(o['genorder_tokens'])
                o_dict['tree_string'] = o['tree_string']
                fname.write(json.dumps(o_dict)+'\n')
    '''with open ('./roc_gen_result/uniform_lm_result_topk_100_gennum_10.txt', 'r') as fname:
        arr = fname.readlines()
        for a in arr:
            a_dict = json.loads(a)
            print(a_dict['tree_string'])'''

def add_arguments(parser):
    parser.add_argument("--topk", type=int, default=100, help="topk sampling.")
    parser.add_argument("--generation_count", type=int, default=10, help="Generate example count.")
    parser.add_argument("--save_dir", type=str, default='./roc_gen_result', help="Result save directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    #python ./evaluate_unconditional.py --topk 100 --generation_count 10 --save_dir ./roc_gen_result
    run(int(args.topk), int(args.generation_count), str(args.save_dir))
