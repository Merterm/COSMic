from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import re

import modeling
import tokenization
import tensorflow as tf

from extract_features import *

def main(_):
	''' Code copied from BERT/extract_features/py '''
	tf.logging.set_verbosity(tf.logging.INFO)

	# 1) Setting up
	layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

	bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

	tokenizer = tokenization.FullTokenizer(
		vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

	is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
	run_config = tf.contrib.tpu.RunConfig(
		master=FLAGS.master,
		tpu_config=tf.contrib.tpu.TPUConfig(
			num_shards=FLAGS.num_tpu_cores,
			per_host_input_for_training=is_per_host))

	# 2) Load the sentences
	examples = read_examples(FLAGS.input_file)

	# 3) Tokenize the sentences
	features = convert_examples_to_features(
		examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

	unique_id_to_feature = {}
	for feature in features:
		unique_id_to_feature[feature.unique_id] = feature

	# 4) Load the model
	print("*_*_*_*_*_* Loading the model *_*_*_*_*_*")
	model_fn = model_fn_builder(
		bert_config=bert_config,
		init_checkpoint=FLAGS.init_checkpoint,
		layer_indexes=layer_indexes,
		use_tpu=FLAGS.use_tpu,
		use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

	# 5) Predict using the model
	# If TPU is not available, this will fall back to normal Estimator on CPU
	# or GPU.
	print("*_*_*_*_*_* Extracting the Features *_*_*_*_*_*")
	estimator = tf.contrib.tpu.TPUEstimator(
		use_tpu=FLAGS.use_tpu,
		model_fn=model_fn,
		config=run_config,
		predict_batch_size=FLAGS.batch_size)

	input_fn = input_fn_builder(
		features=features, seq_length=FLAGS.max_seq_length)

	# 6) Write the output JSON with only CLS feature vector.
	with codecs.getwriter("utf-8")(tf.gfile.Open(FLAGS.output_file,
												"w")) as writer:
		for result in estimator.predict(input_fn, yield_single_examples=True):
			unique_id = int(result["unique_id"])
			feature = unique_id_to_feature[unique_id]
			output_json = collections.OrderedDict()
			output_json["linex_index"] = unique_id
			layer_output = result["layer_output_0"]
			features = [
				round(float(x), 6) for x in layer_output[0:1].flat
			]
			output_json["CLS_features"] = features
			print(("*_*_*_*_*_* Saving Sentence " + str(unique_id) +
			 		" Extracted CLS Features *_*_*_*_*_*"), end="\r")
			writer.write(json.dumps(output_json) + "\n")


if __name__ == "__main__":
	flags.mark_flag_as_required("input_file")
	flags.mark_flag_as_required("vocab_file")
	flags.mark_flag_as_required("bert_config_file")
	flags.mark_flag_as_required("init_checkpoint")
	flags.mark_flag_as_required("output_file")
	tf.app.run()
