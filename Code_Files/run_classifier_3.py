# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
from tensorflow.python.framework import ops
import pandas as pd
import numpy as np
import pickle
import sys
from datetime import datetime

flags = tf.flags

FLAGS = flags.FLAGS

################ CHANGED ################
def del_all_flags(FLAGS):
  """ To clear existing flags when module is re-imported"""
  flags_dict = FLAGS._flags()
  keys_list = [keys for keys in flags_dict]
  for keys in keys_list:
    FLAGS.__delattr__(keys)

del_all_flags(FLAGS) # Clear all flags

# assign_flags contains global declaration of all flag variables
import assign_flags
assign_flags.init()

use_tpu = assign_flags.use_tpu
tpu_name = assign_flags.tpu_name
task_name = assign_flags.task_name
do_train = assign_flags.do_train
do_eval = assign_flags.do_eval
do_predict = assign_flags.do_predict
data_dir = assign_flags.data_dir
vocab_file = assign_flags.vocab_file
bert_config_file = assign_flags.bert_config_file
init_checkpoint = assign_flags.init_checkpoint
max_seq_length = assign_flags.max_seq_length
train_batch_size = assign_flags.train_batch_size
learning_rate = assign_flags.learning_rate
num_train_epochs = assign_flags.num_train_epochs
output_dir = assign_flags.output_dir
apply_rp = assign_flags.apply_rp
pruning_mask = assign_flags.pruning_mask
ckpt_eval_path = assign_flags.ckpt_eval_path
test_labels_known = assign_flags.test_labels_known
weighted_loss = assign_flags.weighted_loss
freeze_bert_weights = assign_flags.freeze_bert_weights
do_train_sample_masks = assign_flags.do_train_sample_masks
do_predict_sample_masks = assign_flags.do_predict_sample_masks
train_file = assign_flags.train_file
test_file = assign_flags.test_file
get_tokenized_text = assign_flags.get_tokenized_text

## Required parameters
flags.DEFINE_string(
    "data_dir", data_dir,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", bert_config_file,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", task_name, "The name of the task to train.")

flags.DEFINE_string("vocab_file", vocab_file,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", output_dir,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "ckpt_eval_path", ckpt_eval_path,
    "The path to the checkpoint directory containing the model to run inference on.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", init_checkpoint,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "train_file", train_file,
    "File name of the test file within the data directory for prediction.")

flags.DEFINE_string(
    "test_file", test_file,
    "File name of the test file within the data directory for prediction.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", max_seq_length,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

############# CHANGED: ADDED 3 NEW FLAGS  #############
flags.DEFINE_bool("test_labels_known", test_labels_known, "Whether the ground truth labels on the test set are known.")

flags.DEFINE_bool("weighted_loss", weighted_loss, "Whether class-weighted loss must be used.")

flags.DEFINE_bool( 
    "freeze_bert_weights", freeze_bert_weights, 
    "Whether to avoid training bert layers and only train pruning mask.")

flags.DEFINE_bool("do_train", do_train, "Whether to run training.")

flags.DEFINE_bool("do_eval", do_eval, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", do_predict,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
  "do_train_sample_masks", do_train_sample_masks, 
  "Whether to train masks for individual samples to extract sample-specific critical subnetworks.")

flags.DEFINE_bool(
  "do_predict_sample_masks", do_predict_sample_masks, 
  "Whether to predict results on sample-specific critical subnetworks.")

flags.DEFINE_bool(
  "get_tokenized_text", get_tokenized_text, 
  "Whether to write the tokenized text to a file")

flags.DEFINE_integer("train_batch_size", train_batch_size, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", learning_rate, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", num_train_epochs,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 200,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 114,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", use_tpu, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("apply_rp", apply_rp, "Whether to apply random pruning or not.")

######## CHANGED: pruning_mask is not a flag any more ########
# flags.DEFINE_string("pruning_mask", pruning_mask, "Random pruning mask to be applied on encoder self attention.")

tf.flags.DEFINE_string(
    "tpu_name", tpu_name,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

FLAGS.use_tpu = False

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, FLAGS.train_file)), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, FLAGS.test_file)), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[1])
      text_b = tokenization.convert_to_unicode(line[2])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[3])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, FLAGS.train_file)), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, FLAGS.test_file)), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      num1 = 1
      num2 = 2
      print("Column indices used are:", num1, num2)
      text_a = tokenization.convert_to_unicode(line[num1])
      text_b = tokenization.convert_to_unicode(line[num1])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[3])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class RteProcessor(DataProcessor):
  """Processor for the RTE data set."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, FLAGS.train_file)), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, FLAGS.test_file)), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[1])
      text_b = tokenization.convert_to_unicode(line[2])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[3])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

class AGNewsProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, FLAGS.train_file)), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, FLAGS.test_file)), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1", "2", "3"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[1])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[2])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    # if FLAGS.do_train_sample_masks:
    #   return self._create_examples(
    #     self._read_tsv(os.path.join(data_dir, "Train_Adv_Examples_SST_Bert_Uncased.tsv")), "train")
    #     # self._read_tsv(os.path.join(data_dir, "train_sample_masks_data2.tsv")), "train")
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, FLAGS.train_file)), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev") ################### CHANGE BACK

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, FLAGS.test_file)), "test") ################### CHANGED THE TEST FILE

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets. [Modified for the SST-2 Dataset]"""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if i == 0: ######################################################### All files in SST-2 have a header column
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1]) #########################################################
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[1]) #########################################################
        label = tokenization.convert_to_unicode(line[2])  #########################################################
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


######################### Processor for News Category Classification task ################################
class NewsProcessor(DataProcessor):
  """Processor for the News Categorization data set."""

  def get_train_examples(self, data_dir):
    """See base class."""
    # if FLAGS.do_train_sample_masks:
    #   return self._create_examples(
    #     self._read_tsv(os.path.join(data_dir, "test.tsv")), "train")
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, FLAGS.train_file)), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, FLAGS.test_file)), "test") 

  def get_labels(self):
    """See base class."""
    return ["0", "1", "2", "3", "4"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0: ##################### All files have a header column
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[1])
        label = tokenization.convert_to_unicode(line[2])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  if FLAGS.get_tokenized_text:
    return tokens

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  tokenized_text = []
  writer = tf.python_io.TFRecordWriter(output_file)
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)
    
    tokenized_text.append(feature)
    if ex_index == len(examples)-1 and FLAGS.get_tokenized_text:
      with open(FLAGS.data_dir+'/outputs/tokenized_test.txt', "wb") as fp:
          pickle.dump(tokenized_text, fp)
      writer.close()
      sys.exit()

    elif not FLAGS.get_tokenized_text:
      def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

      features = collections.OrderedDict()
      features["input_ids"] = create_int_feature(feature.input_ids)
      features["input_mask"] = create_int_feature(feature.input_mask)
      features["segment_ids"] = create_int_feature(feature.segment_ids)
      features["label_ids"] = create_int_feature([feature.label_id])
      features["is_real_example"] = create_int_feature(
          [int(feature.is_real_example)])

      tf_example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings,apply_rp,pruning_mask):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      apply_rp=apply_rp,
      pruning_mask=pruning_mask)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    if FLAGS.weighted_loss:
      label_weights = [[1.0, 0.0], [0.0, 3.0]]
      print("Loss function uses the label weight matrix: ", str(label_weights))
      label_weights = tf.constant(label_weights)
      one_hot_labels = tf.matmul(one_hot_labels, label_weights)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1) 
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


# web_paths = ['https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip']
# bert_in_checkpoint = Dataset.File.from_files(path=web_paths)
# import zipfile
# with zipfile.ZipFile(bert_in_checkpoint, 'r') as zip_ref:
#     zip_ref.extractall('.')


from urllib.request import urlopen
from zipfile import ZipFile

if not os.path.exists('checkpoints/uncased_L-12_H-768_A-12'):
  zipurl = 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'
      # Download the file from the URL
  zipresp = urlopen(zipurl)
      # Create a new file on the hard drive
  tempzip = open("uncased_L-12_H-768_A-12.zip", "wb")
      # Write the contents of the downloaded file into the new file
  tempzip.write(zipresp.read())
      # Close the newly-created file
  tempzip.close()
      # Re-open the newly-created file with ZipFile()
  zf = ZipFile("uncased_L-12_H-768_A-12.zip")
      # Extract its contents into <extraction_path>
      # note that extractall will automatically create the path
  zf.extractall(path = 'checkpoints/')
      # close the ZipFile instance
  zf.close()

  print("Listing files in checkpoints/ :")
  print(os.listdir("checkpoints/"))
  print("Checkpoint file extracted successfully!")

else:
  print("Checkpoint files were already present!")


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings,apply_rp,pruning_mask):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    print("model_fn_flag")
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings,apply_rp,pruning_mask)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, FLAGS.freeze_bert_weights)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features

# def set_global_step_to_last_ckpt(estimator,i):
#   # set global step tensor
#   graph = ops.get_default_graph()
#   if i > 0:
#     ckpt_step_value = int(estimator.latest_checkpoint().split("-")[-1])+1
#     graph.clear_collection(GLOBAL_STEP_READ_KEY)
#     graph.add_to_collection(GLOBAL_STEP_READ_KEY, ckpt_step_value)

#   # read and check value
#   global_step_read_tensors = graph.get_collection(GLOBAL_STEP_READ_KEY)
#   if len(global_step_read_tensors) > 1:
#    raise RuntimeError('There are multiple items in collection {}.There should be only one.'.format(GLOBAL_STEP_READ_KEY))
#   if len(global_step_read_tensors) == 1:
#     print("flag_set_global_step : {}".format(i))
#     print(global_step_read_tensors[0])


def main(_):
  tf.logging.set_verbosity(tf.logging.ERROR)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "news": NewsProcessor,
      "rte" : RteProcessor,
      "ag_news" : AGNewsProcessor
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and \
          not FLAGS.do_train_sample_masks and not FLAGS.do_predict_sample_masks:
    raise ValueError(
        "At least one of `do_train`, `do_eval`, `do_predict', `do_train_sample_masks`\
           or `do_predict_sample_masks` must be True.")

  if FLAGS.do_train_sample_masks and not FLAGS.freeze_bert_weights:
    raise ValueError(
        "`freeze_bert_weights` must be True when `do_train_sample_masks` is True.\
          When training masks for individual samples, conventional BERT layer weights must be frozen.\
          Only pruning mask layers must be trained.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    print("Entered the use_tpu if condition!")
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=100,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None

  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  ############# CHANGED: Training sample-specific masks #############
  if FLAGS.do_train_sample_masks:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(1 / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      apply_rp=FLAGS.apply_rp,
      pruning_mask=assign_flags.pruning_mask)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    # set_global_step_to_last_ckpt(estimator,i)
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training*****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    print("Num steps = ", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    # print("These are trainable variables:")
    # print(tf.trainable_variables())
    # print([v.name for v in tf.trainable_variables()])
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  ############# CHANGED: Training sample-specific masks #############
  if FLAGS.do_train_sample_masks:
    num_hidden_layers = 12 # Based on BERT Model Architecture
    num_heads_per_layer = 12 # Based on BERT Model Architecture
    num_samples_per_iter = 1
    all_sample_masks = []
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    os.system('rm -r '+ FLAGS.output_dir+'*')
    # range(0, 91, num_samples_per_iter):

    for example_num in range(200, min(501,len(train_examples)), num_samples_per_iter):
    # for example_num in range(0, len(train_examples), num_samples_per_iter):
      file_based_convert_examples_to_features(
          train_examples[example_num:example_num+num_samples_per_iter], label_list, 
                    FLAGS.max_seq_length, tokenizer, train_file)
      # tf.logging.info("***** Running training*****")
      # tf.logging.info("  Num examples = %d", len(single_example))
      # tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
      # tf.logging.info("  Num steps = %d", num_train_steps)
      train_input_fn = file_based_input_fn_builder(
          input_file=train_file,
          seq_length=FLAGS.max_seq_length,
          is_training=True,
          drop_remainder=True)
      estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

      # Extract control gate values (i.e., pruning mask layer weights) after
      # training the network on a single sample. The weights across all
      # layers for all samples are stored in all_sample_masks and saved to output
      curr_sample_mask = []
      for layer_num in range(num_hidden_layers):
        curr_layer_mask = estimator.get_variable_value("bert/encoder/layer_"+str(layer_num)+"/attention/self/pruning_mask_layer_name:0")
        curr_layer_mask = list(np.reshape(curr_layer_mask, num_heads_per_layer))
        curr_sample_mask.append(curr_layer_mask)
      all_sample_masks.append(curr_sample_mask)

      # Delete current checkpoints so that they are not restored in the next iteration with new samples
      os.system('rm -r '+ FLAGS.output_dir+'*')
      if example_num%1==0:
        print("{} of {} sample-specific masks trained".format(example_num, len(train_examples)))
    all_sample_masks = np.array(all_sample_masks)
    file_num = 0
    # Ensure that previously created files are not overwritten
    while 'all_sample_masks_'+str(file_num) in os.listdir(FLAGS.data_dir+'/outputs/'):
      file_num += 1
    np.save(FLAGS.data_dir+'/outputs/all_sample_masks_'+str(file_num), all_sample_masks)

# Note: Modify thresholding rule so that there is at least one 1 in each layer, otherwise there will be no info transfer

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

      print("====== eval examples are : =======")
      print(len(eval_examples))
      print("eval batch size :")
      print(FLAGS.eval_batch_size)
      print("eval_steps :")
      print(eval_steps)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, checkpoint_path=FLAGS.ckpt_eval_path)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results*****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn,checkpoint_path=FLAGS.ckpt_eval_path)
    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")

    if FLAGS.test_labels_known:
      true_label_list = pd.read_csv(os.path.join(FLAGS.data_dir, FLAGS.test_file), sep="\t")['label'].to_list()

    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      num_correct = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        true_label_data = ""
        if FLAGS.test_labels_known:
          class_pred = int(np.argmax(probabilities))
          true_label = true_label_list[i]
          if FLAGS.task_name == "mnli":
            label_dict = {"entailment":1, "neutral":2, "contradiction":0}
            true_label = label_dict[true_label]
          num_correct += int(class_pred == true_label)
          true_label_data = "\t" + str(i) + "\t" + str(class_pred) + "\t" + str(true_label)
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + true_label_data + "\n"
        writer.write(output_line)
        num_written_lines += 1
      if FLAGS.test_labels_known:
        print("Test accuracy = ", num_correct/len(true_label_list))
    assert num_written_lines == num_actual_predict_examples

  ########### CHANGED: Predicting on sample-specific masks ###########
  if FLAGS.do_predict_sample_masks:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)

    # num_actual_predict_examples = 1
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample()) 
    true_label_list = pd.read_csv(os.path.join(FLAGS.data_dir, FLAGS.test_file), sep="\t")['label'].to_list()
    mask_file = FLAGS.data_dir+'/outputs/all_sample_masks_hard_concrete_test_first_501.npy'
    all_sample_masks = np.load(mask_file)
    print("Using the masks file: ", mask_file)
    diff = all_sample_masks - 0.5
    thresh_fraction = 0.8
    print("Using threshold of ", thresh_fraction)
    thresh = np.max(diff, axis=(1,2))*thresh_fraction
    all_sample_thresholded_masks = (diff>=np.broadcast_to(np.expand_dims(thresh, axis=(1,2)), diff.shape)).astype(int)
    
    # Ensure that atleast 1 head is active in every layer
    find_zeros = np.all((all_sample_thresholded_masks==0), axis=2)
    for i in range(find_zeros.shape[0]):
        for j in range(find_zeros.shape[1]):
            if find_zeros[i,j]==True:
                row_max_idx = np.argmax(all_sample_masks[i,j])
                all_sample_thresholded_masks[i,j,row_max_idx]=1
    np.save(FLAGS.data_dir+'/outputs/all_sample_thresholded_masks', all_sample_thresholded_masks)

    num_samples_per_iter = 1
    probabilities_list = []
    num_correct = 0
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    predict_drop_remainder = True if FLAGS.use_tpu else False
    os.system('rm -r '+ FLAGS.output_dir+'*')
    num_iterations = 0
    # for example_num in range(400, 600-num_samples_per_iter+1, num_samples_per_iter):
    for example_num in range(0, min(501,len(predict_examples)), num_samples_per_iter):
    # for example_num in range(0, len(predict_examples), num_samples_per_iter):
      # A training checkpoint has to be created with the weights of the pruning
      # mask layer set to the required mask configuration

      model_fn = model_fn_builder(
          bert_config=bert_config,
          num_labels=len(label_list),
          init_checkpoint=FLAGS.init_checkpoint,
          learning_rate=FLAGS.learning_rate,
          num_train_steps=1, #### CHANGED
          num_warmup_steps=0, #### CHANGED
          use_tpu=FLAGS.use_tpu,
          use_one_hot_embeddings=FLAGS.use_tpu,
          apply_rp=FLAGS.apply_rp,
          pruning_mask=str(all_sample_thresholded_masks[num_iterations].tolist()))

      estimator = tf.contrib.tpu.TPUEstimator(
          use_tpu=FLAGS.use_tpu,
          model_fn=model_fn,
          config=run_config,
          train_batch_size=FLAGS.train_batch_size,
          eval_batch_size=FLAGS.eval_batch_size,
          predict_batch_size=FLAGS.predict_batch_size)

      # Perform prediction with the generated checkpoint. Prediction 
      # per `num_samples_per_iter` samples is performed
      file_based_convert_examples_to_features(
          predict_examples[example_num:example_num+num_samples_per_iter], label_list,
                    FLAGS.max_seq_length, tokenizer, predict_file)

      predict_input_fn = file_based_input_fn_builder(
          input_file=predict_file,
          seq_length=FLAGS.max_seq_length,
          is_training=False,
          drop_remainder=predict_drop_remainder)

      result = estimator.predict(input_fn=predict_input_fn,checkpoint_path=FLAGS.ckpt_eval_path)

      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"].tolist()
        class_pred = int(np.argmax(probabilities))
        true_label = true_label_list[example_num+i]
        print("Prediction of critical subnetwork for sample {} is {} which is {} with probability {}".format(example_num+i, class_pred, (class_pred == true_label), np.max(probabilities)))
        probabilities.extend([class_pred, true_label])
        probabilities_list.append(probabilities)
        if class_pred == true_label:
          num_correct+=1

      os.system('rm -r '+ FLAGS.output_dir+'*')
      num_iterations += 1

    # Write final predictions to output file and print overall accuracy
    print("Overall accuracy on critical subnetwork prediction = {}".format(num_correct/len(probabilities_list)))
    output_predict_file = os.path.join(FLAGS.output_dir, "sample_masks_test_results.tsv")

    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      num_correct = 0
      for sample_result in probabilities_list:
        output_line = "\t".join([str(i) for i in sample_result]) + "\n"
        writer.write(output_line)
        num_written_lines += 1

if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
