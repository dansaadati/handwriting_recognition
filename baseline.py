"""
Baseline model for unconstrained offline handwriting recognition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import re

import json

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile



import numpy as np

from os.path import join as pjoin


import logging
import io

from PIL import Image

import collections

IMAGE_HEIGHT = 600
IMAGE_WIDTH = 200

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

"""
	Initialize vocabulary mapping from words in IAM dataset to one-hot encodings
	with frequency ordering
"""

def create_vocab():
	label_data = np.load('word_label_data.npy').item()
	vocab = collections.Counter()

	for img_path in os.listdir("words"):
		img = np.array(Image.open("words/" + img_path))
		
		label_id = os.path.splitext(img_path)[0]
		label = label_data[label_id]
		vocab[label] += 1

	vocab_sorted = {}
	counter = 0
	for word in vocab.most_common():
		counter += 1
		vocab_sorted[word] = counter

   		
	np.save('vocab.npy', vocab)


"""
    Convert IAM dataset into standard Tensorflow TFRecord form
"""
def create_iam_tf_dataset():
	# load label data from parsed xml
	label_data = np.load('word_label_data.npy').item()
	vocab = np.load('vocab.npy').item()

	tfrecords_filename = 'iam.tfrecords'

	writer = tf.python_io.TFRecordWriter(tfrecords_filename)



	for img_path in os.listdir("words"):
		img = np.array(Image.open("words/" + img_path))
		
		label_id = os.path.splitext(img_path)[0]
		label = label_data[label_id]
		vocab[label] += 1
		# so we can reconstruct image from 1d variant later
		height = img.shape[0]
		width = img.shape[1]

		img_raw = img.tostring()
		label_raw = vocab[label]

		# create example to write to tfrecord file
		example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
               'width': _int64_feature(width),
               'image_raw': _bytes_feature(img_raw),
               'label_raw': _int64_feature(label_raw)}))
		writer.write(example.SerializeToString())
	writer.close()


def read_and_decode(filename_queue):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,
		features={
			'height': tf.FixedLenFeature([], tf.int64),
			'width': tf.FixedLenFeature([], tf.int64),
			'image_raw': tf.FixedLenFeature([], tf.string),
			'label_raw': tf.FixedLenFeature([], tf.int64)
	})
	image = tf.decode_raw(features['image_raw'], tf.uint8)
	height = tf.cast(features['height'], tf.int32)
	width = tf.cast(features['width'], tf.int32)
	label = tf.cast(features['label_raw'], tf.int32)

	image_shape = tf.stack([height, width, 1])
	image = tf.reshape(image, image_shape)
	image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.int32)
    
    	# Random transformations can be put here: right before you crop images
    	# to predefined size.
	resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                           target_height=IMAGE_HEIGHT,
                                           target_width=IMAGE_WIDTH)
	batch_size = 4
	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3 * batch_size
	example_batch, label_batch = tf.train.shuffle_batch([resized_image, label], 
	batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)
	return example_batch, label_batch
                                                 

   
   	



def main(_):

	
	tfrecords_filename = 'iam.tfrecords'

	filename_queue = tf.train.string_input_producer([tfrecords_filename])
	image, label = read_and_decode(filename_queue)

    # The op for initializing the variables.
	init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

	with tf.Session()  as sess:
    
		sess.run(init_op)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess,coord=coord)

		for i in xrange(10):
			img, lbl = sess.run([image, label])
			print(img[0,300:500,100:200,:])
			print(lbl)	    	    
		coord.request_stop()
		coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
