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
import metrics
import pprint
from modeling import get_shape_list

import datetime

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
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
flags.DEFINE_string("middle_output", "middle_data", "Dir was used to store middle data!")
flags.DEFINE_bool("crf", True, "use crf!")

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

    @classmethod
    def _read_data(cls, input_file):
        """Read a BIO data!"""
        rf = open(input_file, 'r')
        lines = []
        words = []
        labels = []

        for line in rf:

            '1.取得单词和标签'
            word = line.strip().split(' ')[0]#即使为空字符串，也能取到0
            label = line.strip().split(' ')[-1]
            # here we dont do "DOCSTART" check

            '2.句子结束，将临时存储的word，以空格连接成一个句子，label同样操作。'
            if len(line.strip()) == 0 and words[-1] == '.':
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append((l, w))
                words = []
                labels = []
                continue
            words.append(word)
            labels.append(label)
        rf.close()
        return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        with tf.gfile.GFile(os.path.join(data_dir, "text.txt")) as f:
            text_list=f.readlines()
        with tf.gfile.GFile(os.path.join(data_dir, "predicate.txt"))as f:
            predicate_list=f.readlines()
        with tf.gfile.GFile(os.path.join(data_dir, "tag.txt"))as f:
            seqtag_list=f.readlines()
        return self._create_example(text_list,predicate_list,seqtag_list, "train"
        )

    def get_dev_examples(self, data_dir):
        with tf.gfile.GFile(os.path.join(data_dir, "text.txt")) as f:
            text_list = f.readlines()
        with tf.gfile.GFile(os.path.join(data_dir, "predicate.txt"))as f:
            predicate_list = f.readlines()
        with tf.gfile.GFile(os.path.join(data_dir, "tag.txt"))as f:
            seqtag_list = f.readlines()
        return self._create_example(text_list, predicate_list, seqtag_list, "dev"
                                    )
    def get_test_examples(self, data_dir):
        with tf.gfile.GFile(os.path.join(data_dir, "text.txt")) as f:
            text_list = f.readlines()
        with tf.gfile.GFile(os.path.join(data_dir, "predicate.txt"))as f:
            predicate_list = f.readlines()
        with tf.gfile.GFile(os.path.join(data_dir, "tag.txt"))as f:
            seqtag_list = f.readlines()
        return self._create_example(text_list, predicate_list, seqtag_list, "test"
                                    )



    def get_labels(self):
        """
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        :return:
        """

#        return ["[PAD]", "B-SUB", "I-SUB", "O", "B-OBJ", "I-OBJ","X", "[CLS]","[SEP]","[category]"]
        return ["[PAD]", "B-SUB", "O", "B-OBJ", "X", "[CLS]", "[SEP]", "[category]"]

    def _create_example(self, text_list,predicate_list,seqtag_list, set_type):
        '''
        :type text_list:List[str]
        :type predicate_list:List[str]
        :type seqtag_list:List[str]
        :return:
        '''
        examples = []
        for i in range(len(text_list)):
            guid = "%s-%s" % (set_type, i)
            text_a=tokenization.convert_to_unicode(text_list[i].strip())
            text_b=tokenization.convert_to_unicode(predicate_list[i].strip())

            label=seqtag_list[i].strip().replace("I-OBJ","B-OBJ")
            label=label.replace("I-SUB","B-SUB")

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

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
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

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
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            if set_type == "test":
                label = "contradiction"
            else:
                label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

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
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[1])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[3])
                label = tokenization.convert_to_unicode(line[1])
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
            segment_ids=[1] * max_seq_length,
            label_id=[0] * max_seq_length,
            is_real_example=False),["[PAD]"] * max_seq_length,[0] * max_seq_length
    '---------1.将lable中的标签替换为id(ok)-----------'
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    predicate_map={}
    predicate_list=['Instrument-Agency', 'Product-Producer', 'Content-Container', 'Entity-Destination', 'Cause-Effect', 'Member-Collection', 'Component-Whole', 'Other', 'Entity-Origin', 'Message-Topic']
    for (i,predicate) in enumerate(predicate_list):
        predicate_map[predicate]=i+1#为了避免与pad重叠

    '--------2.对于原英文单词分为2个单词的标记为不同标签，后缀词为×--------'
    textlist = example.text_a.split(' ')  # 对text_A进行空格区分
    labellist = example.label.split(' ')  # 对标签进行空格区分
    tokens = []
    labels = []
    for i, (word, label) in enumerate(zip(textlist, labellist)):
        token = tokenizer.tokenize(word)  # 对
        tokens.extend(token)
        for i, _ in enumerate(token):
            if i == 0:  # 第一个单词的tag为正常
                labels.append(label)
            else:  # 被拆分为第2个单词及以后的tag为×
                labels.append("X")

    '--------3.利用残差的原理，构建和text_a一样长度的text_b-----'
    tokens_b = None
    if example.text_b:
        tokens_b=[example.text_b]*len(tokens)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 1)]
            labels=labels[0:(max_seq_length - 1)]

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
    '------4.遍历tokens列表，都转化为id,不足长度补0---'
    '--------4.1 遍历tokens列表，label_ids转化为id-------'
    ntokens = []
    segment_ids = []
    label_ids=[]
    ntokens.append("[CLS]")
    segment_ids.append(1)
    label_ids.append(label_map["[CLS]"])

    for i,token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(1)
        label_ids.append(label_map[labels[i]])

    '-----------4.2 添加SEP---'
    ntokens.append("[SEP]")
    segment_ids.append(1)
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)


    '-----------4.3 添加tokens_b中的字符--'

    if tokens_b:
        for token in tokens_b:
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map["[category]"])
            input_ids.append(predicate_map[example.text_b])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        input_ids.extend(tokenizer.convert_tokens_to_ids(["[SEP]"]))

    '-----------4.4 不足长度补0---'


    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(1)
        label_ids.append(0)
        ntokens.append("[PAD]")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    '-------------5.构造Feature,将label_id赋值为label_ids-----------'
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_ids,
        is_real_example=True)
    return feature, ntokens, label_ids


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)
    batch_tokens = []
    batch_labels = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        a=0
        c=a+2
        feature , ntokens, label_ids= convert_single_example(ex_index, example, label_list,max_seq_length, tokenizer)

        batch_tokens.extend(ntokens)
        batch_labels.extend(label_ids)
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_id)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()
    return batch_tokens, batch_labels


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
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
                 labels, num_labels, use_one_hot_embeddings,crf):
    """Creates a classification model."""
    '---------------1.获得模型的序列输出--'
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_sequence_output()#[batch_size, seq_length, hidden_size]


    with tf.variable_scope("loss"):
        '-----2.添加全连接层，计算损失函数--------'

        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.layers.dense(output_layer,num_labels)
        if crf:
            '--------------3 采用crf的损失函数--'
            mask2len = tf.reduce_sum(segment_ids, axis=1)
            log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(logits, labels, mask2len)
            loss = tf.math.reduce_mean(-log_likelihood)
            predict, viterbi_score = tf.contrib.crf.crf_decode(logits, transition, mask2len)
            return (loss, logits, predict)
        else:
            '--------------4 softmax损失函数------'
            '-----4.1 batch_size*seq_len个样本的分类问题---------'
            logits = tf.reshape(logits, [-1, num_labels])
            labels=tf.reshape(labels,[-1])
            mask=tf.cast(input_mask,dtype=tf.float32)#由原来的int转化为float
            seg_mask=tf.cast(segment_ids,dtype=tf.float32)
            '4.2 设计weight值为mask部分，调用softmax_cross_entropy函数'
            loss=tf.losses.sparse_softmax_cross_entropy(logits=logits,
                                                        labels=labels,
                                                        weights=tf.reshape(seg_mask, [-1]))
            '-----4.3 计算概率和预测--------'
            probabilities = tf.math.softmax(logits, axis=-1)
            predict = tf.math.argmax(probabilities, axis=-1)



        return (loss, logits,predict)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings,crf):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

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

        (total_loss, logits, predicts) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings,crf=crf)

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
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, logits,input_mask,segment_ids ):
                label_ids=tf.reshape(label_ids, [-1])
                num_labels=get_shape_list(logits)[-1]
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)#logits是32,128,12
                temp=[1,3,4]

                precision = metrics.precision(label_ids, predictions, num_labels,pos_indices=temp, weights=segment_ids)
                recall = metrics.recall(label_ids, predictions, num_labels, pos_indices=temp,weights=segment_ids)
                f2 = metrics.fbeta(label_ids, predictions, num_labels, pos_indices=temp,weights=segment_ids, beta=2)
                f1 = metrics.f1(label_ids, predictions, num_labels, pos_indices=temp,weights=segment_ids)
                eval_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'f2': f2
                }
                return eval_metrics
            test_predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            test_labels_ids=tf.reshape(label_ids, [-1])
            eval_metrics = (metric_fn, [label_ids, logits,input_mask,segment_ids])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            predict_dict = {'predictions': predicts}
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predict_dict,
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
def _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i):
    token = batch_tokens[i]
    predict = id2label[prediction]
    true_l = id2label[batch_labels[i]]
    # token是正常符号（不是Pad和cls），跳过原来被分为##的字符。会缺少一些字符。
    if token != "[PAD]" and token != "[CLS]" and true_l != "X":
        #
        if predict == "X" and not predict.startswith("##"):  # 如果预测为×的且字符开头不是##，不计算.这里predict标签本来就不含有##
            predict = "O"
        line = "{}\t{}\t{}\n".format(token, true_l, predict)
        wf.write(line)


def Writer(output_predict_file, result, batch_tokens, batch_labels, id2label,crf):
    with tf.gfile.GFile(output_predict_file, 'w') as wf:

        if crf:  # crf返回的是一个嵌套的list，另一个不是
            predictions = []
            for m, pred in enumerate(result):
                predictions.extend(pred)
            for i, prediction in enumerate(predictions):
                _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i)

        else:
            for i, prediction in enumerate(result):
                _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i)
def load_global_step_from_checkpoint_dir(checkpoint_dir):
    try:
        checkpoint_reader = tf.train.NewCheckpointReader(
            tf.train.latest_checkpoint(checkpoint_dir))
        return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
    except:  # pylint: disable=bare-except
        return 0

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "xnli": XnliProcessor,
        "ner":NerProcessor
    }

    TASK_DATA_DIR = '/content/data4relation_2/'
    '--------------1.设置Bert_config,------------------'


    BERT_MODEL = 'uncased_L-12_H-768_A-12'
    BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL
    CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
    INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
    VOCAB_FILE=os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
    OUTPUT_DIR = 'gs://storage_hx/test/models/NER4Relation/data4relation_2_2lable'

    CRF=False
    DO_TRAIN=False
    DO_EVAL=False
    DO_PREDICT=True
    DO_LOWER_CASE = True  # uncased，True;cased,False

    USE_TPU = True
    TPU_ADDRESS = 'grpc://10.101.14.242:8470'
    with tf.Session(TPU_ADDRESS) as session:
        print('TPU devices:')
        pprint.pprint(session.list_devices())
    NUM_TPU_CORES = 8
    ITERATIONS_PER_LOOP = 200
    STEPS_PER_EVAL=200

    TRAIN_BATCH_SIZE = 64
    EVAL_BATCH_SIZE = 64
    PREDICT_BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 8.0
    MAX_SEQ_LENGTH = 64
    # Warmup is a period of time where hte learning rate
    # is small and gradually increases--usually helps training.
    WARMUP_PROPORTION = 0.1
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 1000  # 原来为1000，现在更改为100
    SAVE_SUMMARY_STEPS = 500
    bert_config = modeling.BertConfig.from_json_file(CONFIG_FILE)
    tokenization.validate_case_matches_checkpoint(DO_LOWER_CASE,INIT_CHECKPOINT)

    tf.gfile.MakeDirs(OUTPUT_DIR)
    if MAX_SEQ_LENGTH > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (MAX_SEQ_LENGTH, bert_config.max_position_embeddings))


    '----------2.设置processor,label_list,tokenizer，run_config------------------------'
    processor = processors['ner']()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
    print(tokenizer.tokenize("This here's an example of using the BERT tokenizer"))

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
    if USE_TPU is False:
        tpu_cluster_resolver = None

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=OUTPUT_DIR,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,  # 含义参照Estimator
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=ITERATIONS_PER_LOOP,
            num_shards=NUM_TPU_CORES,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    '--------------3.如果是训练的话，processor获得训练的例子------------------'
    if DO_TRAIN:

        train_examples = processor.get_train_examples(TASK_DATA_DIR+'train/')
        num_train_steps = int(
            len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
        num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)


    '---------------4.设置model_fn,赋值给TPUEstimator--------------'
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=INIT_CHECKPOINT,
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=USE_TPU,
        use_one_hot_embeddings=USE_TPU,
        crf=CRF
    )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=USE_TPU,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        predict_batch_size=PREDICT_BATCH_SIZE)


    if DO_TRAIN and DO_EVAL:
        '------------------1.准备train 的数据集-'
        print('***** Started training at {} *****'.format(datetime.datetime.now()))
        print('  Num examples = {}'.format(len(train_examples)))
        print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
        tf.logging.info("  Num steps = %d", num_train_steps)
        '----处理train_examples,并以TFRecord形势存储文件---------------'
        '-------------将获得train_examples用于------------'
        # 从output_dir中找到需要训练的文件，
        train_file = os.path.join(OUTPUT_DIR, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, MAX_SEQ_LENGTH, tokenizer, train_file)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=MAX_SEQ_LENGTH,
            is_training=True,
            drop_remainder=True)

        '-----------------2.准备eval的数据集------'
        eval_examples = processor.get_dev_examples(TASK_DATA_DIR + 'eval/')
        num_actual_eval_examples = len(eval_examples)
        if USE_TPU:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % EVAL_BATCH_SIZE != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(OUTPUT_DIR, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", EVAL_BATCH_SIZE)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if USE_TPU:
            assert len(eval_examples) % EVAL_BATCH_SIZE == 0
            eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
        eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
        eval_drop_remainder = True if USE_TPU else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        '=================train_and_eval==========='
        current_step = load_global_step_from_checkpoint_dir(OUTPUT_DIR)
        steps_per_eval = STEPS_PER_EVAL
        while current_step < num_train_steps:
            '---train---------------'
            next_checkppint = min(current_step + steps_per_eval, num_train_steps)
            estimator.train(input_fn=train_input_fn, max_steps=int(next_checkppint))
            current_step = next_checkppint

            '----eval---------------'
            print('***** Started evaluation at {} *****'.format(datetime.datetime.now()))
            print('  Num examples = {}'.format(len(eval_examples)))
            print('  Batch size = {}'.format(EVAL_BATCH_SIZE))
            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
            print('***** Finished evaluation at {} *****'.format(datetime.datetime.now()))
            output_eval_file = os.path.join(OUTPUT_DIR, "eval_results.txt")
            with tf.gfile.GFile(output_eval_file, "w") as writer:
                print("***** Eval results *****")
                for key in sorted(result.keys()):
                    print('  {} = {}'.format(key, str(result[key])))
                    writer.write("%s = %s\n" % (key, str(result[key])))



    '-----------------5.训练------------------'
    if DO_TRAIN and not DO_EVAL:
        '------------------1.准备train 的数据集-'
        print('***** Started training at {} *****'.format(datetime.datetime.now()))
        print('  Num examples = {}'.format(len(train_examples)))
        print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
        tf.logging.info("  Num steps = %d", num_train_steps)
        '----处理train_examples,并以TFRecord形势存储文件---------------'
        '-------------将获得train_examples用于------------'
        # 从output_dir中找到需要训练的文件，
        train_file = os.path.join(OUTPUT_DIR, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, MAX_SEQ_LENGTH, tokenizer, train_file)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=MAX_SEQ_LENGTH,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    '----------------6.评测------'
    if DO_EVAL and not DO_TRAIN:
        eval_examples = processor.get_dev_examples(TASK_DATA_DIR + 'eval/')
        num_actual_eval_examples = len(eval_examples)
        if USE_TPU:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % EVAL_BATCH_SIZE != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(OUTPUT_DIR, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", EVAL_BATCH_SIZE)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if USE_TPU:
            assert len(eval_examples) % EVAL_BATCH_SIZE == 0
            eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
        eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
        eval_drop_remainder = True if USE_TPU else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        '------------------将评测结果写入评测文件-----------'
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(OUTPUT_DIR, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


    '-----------------7.预测---------------'
    if DO_PREDICT:
        predict_examples = processor.get_test_examples(TASK_DATA_DIR + 'predict/')
        num_actual_predict_examples = len(predict_examples)
        if USE_TPU:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % PREDICT_BATCH_SIZE != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(OUTPUT_DIR, "predict.tf_record")
        batch_tokens, batch_labels=file_based_convert_examples_to_features(predict_examples, label_list,
                                                MAX_SEQ_LENGTH, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", PREDICT_BATCH_SIZE)

        predict_drop_remainder = True if USE_TPU else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        result = list(result)
        result = [pred['predictions'] for pred in result]
        label_list = processor.get_labels()
        id2label = {}
        for (i, label) in enumerate(label_list):
            id2label[i] = label

        output_predict_file = os.path.join('/content/data4relation_2/predict', "test_results.tsv")
        Writer(output_predict_file, result, batch_tokens, batch_labels, id2label,crf=CRF)
        print('ok')


if __name__ == "__main__":
    tf.app.run()

