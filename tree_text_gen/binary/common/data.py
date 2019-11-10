from torch.utils.data import Dataset

import jsonlines
import torch as th


def load_personachat(filepath, log=True):
    with open(filepath, 'r', encoding='utf-8') as f:
        with jsonlines.Reader(f) as reader:
            data = [line for line in reader]
    if log:
        print("%d sentences" % (len(data)))
    return data


def build_tok2i(tokens):
    def _add(t, d):
        if t not in d:
            d[t] = len(d)
    tok2i = {}
    for tok in ['<s>', '<p>', '</s>', '<unk>', '<end>'] + tokens:
        _add(tok, tok2i)
    return tok2i


def inds2toks(i2tok, inds_list):
    toks = [i2tok.get(i, '<unk>') for i in inds_list]
    return toks


class SentenceDataset(Dataset):
    def __init__(self, data, tok2i, max_tokens=-1):
        x_data = [d for d in data if max_tokens == -1 or (len(d['x_tokens']) <= max_tokens)]
        y_data = [d for d in data if max_tokens == -1 or (len(d['y_tokens']) <= max_tokens)]
        #{'tokens': ['i', 'am', 'doing', 'great', 'except', 'for', 'the', 'allergies', '.']}
        self.tok2i = tok2i
        self.i2tok = {j: i for i, j in tok2i.items()}
        # pre-initialize token-index vectors
        for d in data:
            # indices of: t1 t2 ... tT </s>
            d['y_token_idxs'] = ([tok2i.get(t, tok2i['<unk>']) for t in d['y_tokens']] +
                               [tok2i['</s>']])
            d['x_token_idxs'] = ([tok2i.get(t, tok2i['<unk>']) for t in d['x_tokens']] +
                               [tok2i['</s>']])

        self.data = data
        
    def collate(self, batch):
        # batch: length B list of dicts
        # X:     B x Tmax token indices; t1, t2, ..., tT, </s>, <p>, ..., <p>
        x_tmax = max([len(d['x_token_idxs']) for d in batch])
        with th.no_grad():
            X = th.zeros(len(batch), x_tmax, dtype=th.long) + self.tok2i['<p>']
            for i, d in enumerate(batch):
                t = len(d['x_token_idxs'])
                X[i, :t] = th.tensor(d['x_token_idxs'], dtype=th.long)

        y_tmax = max([len(d['y_token_idxs']) for d in batch])
        with th.no_grad():
            Y = th.zeros(len(batch), y_tmax, dtype=th.long) + self.tok2i['<p>']
            for i, d in enumerate(batch):
                t = len(d['y_token_idxs'])
                Y[i, :t] = th.tensor(d['y_token_idxs'], dtype=th.long)
        return X, Y, batch

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

