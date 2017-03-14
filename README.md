Language-model baseline for pronoun prediction
==============================================

This is the language model baseline for pronoun prediction, which we
first used at the DiscoMT 2015 shared task. This is the version for
the WMT 2016 task, which supports multiple languages. Each language
has a configuration file (english.yml, french.yml, german.yml)
which can be selected by adding "--conf thatfile.yml" to the command
line.

For space reason, validation data and language models are not included.
Find these at http://data.statmt.org/wmt16/pronoun-task/

Adding a new task / language pair
---------------------------------

The whole LM baseline is driven by a [produce](https://github.com/texttheater/produce) script that specifies how the original dev data will be transformed into derived data (including evaluation.

In particular:
 * the baseline script itself will be used to create xyz.src-tgt.N.predictions.txt
 * based on that, the evaluation script creates an xyz.src-tgt.N.eval.txt

where xyz is the base name, src-tgt is a language pair such as "de-en", and
N is the weight adjustment for the None/Other item.

The baseline script in turn is driven by YAML file named src-tgt.yml (for a
particular language pair of src and tgt).

In the Yaml file:
 * `all_fillers` is a list of fillers that can be predicted as pronouns. These
   fillers are not words, but sequences of words, which means that the whole
   value will be a list of lists.
 * `other_fillers` is a list of word sequences that contribute towards the
   score of an OTHER prediction
 * `lm` contains the filename of a KenLM (trie) language model
 * `removepos` is true if the dev data contains POS tags that are not part
   of the training sequences that the language model has been trained on.


