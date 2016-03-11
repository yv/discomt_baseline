#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import re
import math
import optparse
import kenlm
import yaml
from collections import defaultdict
from gzip import GzipFile

'''
takes a classification file (optionally gzipped) as input
and writes a file with two columns (replacement+target)

Input format:
  - classes (ignored)
  - true replacements (ignored)
  - source text (currently ignored)
  - target text
  - alignments (currently ignored)

Output format (for fmt=replace):
  - predicted classes
  - predicted replacements
  - original source text
  - original target text
  - alignments

The YAML configuration file has the following keys:
  - all_fillers is a list of equivalence classes of fillers
    (i.e., the first item in the list is the "canonical" representer)
  - other_fillers is a flat list of all dummy fillers that will be
    used (in addition to not inserting any word)
  - lm can point to a KenLM language model file that is used if none
    is specified as a command line option.
'''

oparse = optparse.OptionParser(usage='%prog [options] input_file')
oparse.add_option('--conf', dest='conf',
                  help='YAML configuration')
oparse.add_option('--removepos', dest='removepos', action='store_true',
                  default=None,
                  help='ignore POS tags in input')
oparse.add_option('--lm', dest='lm',
                  help='language model file')
oparse.add_option('--fmt', dest='fmt',
                  choices=['replace', 'predicted',
                           'both', 'compare', 'scores'],
                  help='format (replace, predicted, both, compare, scores)',
                  default='replace')
oparse.add_option('--none-penalty', dest='none_penalty',
                  type='float', default=0.0,
                  help='penalty for empty filler')

replace_re = re.compile('REPLACE_[0-9]+')

all_fillers = [
    ['il'], ['elle'],
    ['ils'], ['elles'],
    ["c'"], ["ce"], ["ça"], ['cela'], ["on"]]

non_fillers = [[w] for w in
               '''
               le l' se s' y en qui que qu' tout
               faire ont fait est parler comprendre chose choses
               ne pas dessus dedans
               '''.strip().split()]

def map_from_unicode(fillers):
    #print >>sys.stderr, fillers
    result = []
    for filler in fillers:
        result.append([w.encode('UTF-8') for w in filler])
    return result

def map_class(filler_name):
    if [filler_name] in non_fillers:
        return 'OTHER'
    elif filler_name == 'NONE':
        return 'OTHER'
    elif filler_name == "c'":
        return 'ce'
    else:
        return filler_name

NONE_PENALTY = 0


def gen_items(contexts, prev_contexts):
    '''
    extends the items from *prev_contexts* with
    fillers and the additional bits of context from
    *contexts*

    returns a list of (text, score, fillers) tuples,
    and expects prev_contexts to have the same shape.
    '''
    if len(contexts) == 1:
        return [(x + contexts[0], y, z)
                for (x, y, z) in prev_contexts]
    else:
        #print >>sys.stderr, "gen_items %s %s"%(contexts, prev_contexts)
        context = contexts[0]
        next_contexts = []
        for filler in all_fillers:
            next_contexts += [(x + context + filler, y, z + filler)
                              for (x, y, z) in prev_contexts]
        for filler in non_fillers:
            next_contexts += [(x + context + filler, y, z + filler)
                              for (x, y, z) in prev_contexts]
        #print >>sys.stderr, "x=%s context=%s"%(x, context)
        next_contexts += [(x + context, y + NONE_PENALTY, z + ['NONE'])
                          for (x, y, z) in prev_contexts]
        if len(next_contexts) > 5000:
            print >>sys.stderr, "Too many alternatives, pruning some..."
            next_contexts = next_contexts[:200]
            next_contexts.sort(key=score_item, reverse=True)
        return gen_items(contexts[1:], next_contexts)


def score_item(x):
    model_score = model.score(' '.join(x[0]))
    return model_score + x[1]

pos_re=re.compile(r'\|(?:[A-Za-z\.:]+)')
def strip_pos(parts):
    result = []
    for part in parts:
        result0 = []
        for word in part:
            word = pos_re.subn('', word)[0]
            result0.append(word)
        result.append(result0)
    return result

def main(argv=None):
    global model, NONE_PENALTY
    global all_fillers, non_fillers
    opts, args = oparse.parse_args(argv)
    if not args:
        oparse.print_help()
        sys.exit(1)
    NONE_PENALTY = opts.none_penalty
    if opts.conf is not None:
        print >>sys.stderr, "Reading config from %s"%(opts.conf,)
        conf = yaml.load(file(opts.conf))
        all_fillers = map_from_unicode(conf['all_fillers'])
        non_fillers = map_from_unicode(conf['other_fillers'])
        if 'lm' in conf and opts.lm is None:
            opts.lm = conf['lm']
        if 'removepos' in conf and opts.removepos is None:
            opts.removepos = conf['removepos']
    if opts.lm is None:
        opts.lm = 'corpus.5.fr.trie.kenlm'
    if opts.removepos is None:
        opts.removepos = False
    discomt_file = args[0]
    print >>sys.stderr, "Loading language model..."
    model = kenlm.LanguageModel(opts.lm)
    mode = opts.fmt
    print >>sys.stderr, "Processing stuff..."
    if discomt_file.endswith('.gz'):
        f_input = GzipFile(discomt_file)
    else:
        f_input = file(discomt_file)
    for i, l in enumerate(f_input):
        if l[0] == '\t':
            if mode == 'replace':
                print l,
                continue
            elif mode != 'scores':
                continue
        classes_str, target, text_src, text, text_align = l.rstrip('\n').split(
            '\t')
        if mode == 'scores':
            print '%d\tTEXT\t%s\t%s\t%s' % (i, text_src, text, text_align)
            if l[0] == '\t':
                continue
        text = replace_re.sub('REPLACE', text)
        targets = [x.strip() for x in target.split(' ')]
        classes = [x.strip() for x in classes_str.split(' ')]
        contexts = [x.strip().split() for x in text.split('REPLACE')]
        if opts.removepos:
            contexts = strip_pos(contexts)
        # print "TARGETs:", target
        # print "CONTEXTs: ", contexts
        if len(contexts) > 5:
            print >>sys.stderr, "#contexts:", len(contexts)
        items = gen_items(contexts, [([], 0.0, [])])
        items.sort(key=score_item, reverse=True)
        pred_fillers = items[0][2]
        pred_classes = [map_class(x) for x in pred_fillers]
        if mode == 'scores':
            # TODO compute individual scores for each slot
            # and convert the scores to probabilities
            scored_items = []
            for item in items:
                words, penalty, fillers = item
                scored_items.append((words, score_item(item), fillers))
            best_penalty = max([x[1] for x in items])
            dists = [defaultdict(float) for k in items[0][2]]
            for words, penalty, fillers in scored_items:
                exp_pty = math.exp(penalty - best_penalty)
                for j, w in enumerate(fillers):
                    dists[j][w] += exp_pty
            for j in xrange(len(items[0][2])):
                sum_all = sum(dists[j].values())
                if sum_all == 0:
                    sum_all = 1.0
                items = [(k, v / sum_all) for k, v in dists[j].iteritems()]
                items.sort(key=lambda x: -x[1])
                print "%s\tITEM %d\t%s" % (
                    i, j, ' '.join([
                        '%s %.4f' % (x[0], x[1])
                        for x in items if x[1] > 0.001]))
        elif mode == 'both':
            print "%s\t%s" % (target, ' '.join(pred_fillers))
        elif mode == 'predicted':
            print "%s\t%s" % (
                ' '.join(pred_classes),
                ' '.join(pred_fillers))
        elif mode == 'replace':
            print "%s\t%s\t%s\t%s\t%s" % (
                ' '.join(pred_classes),
                ' '.join(pred_fillers),
                text_src, text, text_align)
        elif mode == 'compare':
            assert len(classes) == len(pred_classes), (classes, pred_classes)
            for gold, syst in zip(classes, pred_classes):
                print "%s\t%s" % (gold, syst)

if __name__ == '__main__':
    main()
