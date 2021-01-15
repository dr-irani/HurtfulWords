#!/h/haoran/anaconda3/bin/python
import pandas as pd
import numpy as np
import os
from pytorch_pretrained_bert.tokenization import BertTokenizer
import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser('''Sentences from the sentence tokenizer can be very short. This script packs together several sentences into sequences
                                 to ensure that tokenA and tokenB have some minimum length (guaranteed except for sentences at the end of a document) when training BERT''')
parser.add_argument("input_loc", help = "pickled dataframe with 'sents' column", type=str)
parser.add_argument('output_loc', help = "path to output the dataframe", type=str)
parser.add_argument("model_path", help = 'folder with trained SciBERT model and tokenizer', type=str)
parser.add_argument("--under_prob", help = 'probability of being under the limit in a sequence', type=float, default = 0)
parser.add_argument('-m','--minlen', help = 'minimum lengths of tokens to pack the sentences into. Note that this is the length of a SINGLE sequence, not both', nargs = '+',
                     type=int,  dest='minlen', default = [20])
args = parser.parse_args()

tqdm.pandas()

tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case = True)

data_path = '/'.join(args.input_loc.split('/')[:-1])
files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if 'extract' in f])

df = pd.read_pickle(files[0])
for f in tqdm(files[1:]):
    df = pd.concat([df, pd.read_pickle(f)])
    print(len(df))

# df = pd.read_pickle(args.input_loc)

def pack_sentences(row, minlen):
    i, cumsum, init = 0,0,0
    seqs, tok_len_sums = [], []
    while i<len(row.sent_toks_lens):
        cumsum += row.sent_toks_lens[i]
        if cumsum>= minlen:
            if init == i or random.random() >= args.under_prob:
                seqs.append('\n'.join(row.sents[init:i+1]))
            else: #roll back one
                seqs.append('\n'.join(row.sents[init:i]))
                cumsum -= row.sent_toks_lens[i]
                i -=1
            tok_len_sums.append(cumsum)
            cumsum = 0
            init = i+1
        i+=1
    if init != i:
        seqs.append('\n'.join(row.sents[init:]))
        tok_len_sums.append(cumsum)
    return [seqs, tok_len_sums]

print(args.minlen)
for i in args.minlen:
    df['BERT_sents'+str(i)], df['BERT_sents_lens'+str(i)] = zip(*df.progress_apply(pack_sentences, axis = 1, minlen = i))
    df['num_BERT_sents'+str(i)] = df['BERT_sents'+str(i)].progress_apply(len)
    assert(all(df['BERT_sents_lens'+str(i)].apply(sum) == df['sent_toks_lens'].apply(sum)))

for i, data in tqdm(enumerate(np.array_split(df, 10))):
    data.to_pickle(f'/media/data_1/darius/data/df_grouped_{i}.pkl')

# df.to_pickle(args.output_loc)
