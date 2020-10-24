# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import matplotlib.pyplot as plt
import json
import os


# input:
#  alignment matrix - numpy array
#  shape (target tokens + eos, number of hidden source states = source tokens +eos)
# one line correpsonds to one decoding step producing one target token
# each line has the attention model weights corresponding to that decoding step
# each float on a line is the attention model weight for a corresponding source state.
# plot: a heat map of the alignment matrix
# x axis are the source tokens (alignment is to source hidden state that roughly corresponds to a source token)
# y axis are the target tokens
# note: input file format-------------------------------------------
# { "0": {
#     "source": " ",       # the source sentence (without <s> and </s> symbols)
#     "translation": " ",      # the target sentence (without <s> and </s> symbols)
#     "attentions": [         # various attention results
#         {
#             "name": " ",        # a unique name for this attention
#             "type": " ",        # the type of this attention (simple or multihead)
#             "value": [...]      # the attention weights, a json array
#               note: Mij: attention of i-th source token to j-th target token
#         },      # end of one attention result
#         {...}, ...]     # end of various attention results
#     },   # end of the first sample
#   "1":{
#         ...
#     },       # end of the second sample
#     ...
# }       # end of file

# http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
def plot_head_map(mma, target_labels, source_labels):
    fig, ax = plt.subplots(figsize=(20, 40))
    heatmap = ax.pcolor(mma, cmap=plt.cm.Blues)

    # put the major ticks at the middle of each cell
    ax.set_xticks(numpy.arange(mma.shape[1]) + 0.5, minor=False)
    ax.set_yticks(numpy.arange(mma.shape[0]) + 0.5, minor=False)

    # without this I get some extra columns rows
    # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
    ax.set_xlim(0, int(mma.shape[1]))
    ax.set_ylim(0, int(mma.shape[0]))

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # source words -> column labels
    ax.set_xticklabels(source_labels, minor=False)
    # target words -> row labels
    ax.set_yticklabels(target_labels, minor=False)

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    plt.savefig('temp.png')


# column labels -> target words
# row labels -> source words

def read_alignment_matrix(f):
    header = f.readline().strip().split('|||')
    if header[0] == '':
        return None, None, None, None
    sid = int(header[0].strip())
    # number of tokens in source and translation +1 for eos
    src_count, trg_count = map(int, header[-1].split())
    # source words
    source_labels = header[3].decode('UTF-8').split()
    # source_labels.append('</s>')
    # target words
    target_labels = header[1].decode('UTF-8').split()
    target_labels.append('</s>')

    mm = []
    for r in range(trg_count):
        alignment = map(float, f.readline().strip().split())
        mm.append(alignment)
    mma = numpy.array(mm)
    return sid, mma, target_labels, source_labels


def read_plot_alignment_matrices(f, start=0):
    attentions = json.load(f, encoding="utf-8")

    for idx, att in attentions.items():
        if int(idx) < start:
            continue
        source_labels = att["source"].split() + ["SEQUENCE_END"]
        target_labels = att["translation"].split()
        att_list = att["attentions"]
        assert att_list[0]["type"] == "simple", "Do not use this tool for multihead attention."
        mma = numpy.array(att_list[0]["value"])
        if mma.shape[0] == len(target_labels) + 1:
            target_labels += ["SEQUENCE_END"]

        plot_head_map(mma, target_labels, source_labels)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', '-i', type=argparse.FileType("rb"),
    #                     default="trans.att",
    #                     metavar='PATH',
    #                     help="Input file (default: standard input)")
    # parser.add_argument('--start', type=int, default=0)
    #
    # args = parser.parse_args()

    with open('grocery_2.attention', 'rb') as infile:
        a_start = 0
        read_plot_alignment_matrices(infile, a_start)

        pass
