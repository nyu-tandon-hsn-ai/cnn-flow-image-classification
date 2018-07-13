# Copyright 2015 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from  netaddr import IPAddress
from ipaddress import IPv4Address
import gzip
import os

import tensorflow.python.platform

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)

    ### kc, changed magic number from 2051 to 2050
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

SESSION_BYTE_LEN=13

def _big_endian_bytes2int(byte_data):
    res = 0
    power = 1
    for each_byte_data in byte_data[::-1]:
        res += power * each_byte_data
        power *= 256
    return int(res)

def _extract_session_info(byte_data):
    assert len(byte_data) == SESSION_BYTE_LEN
    assert byte_data[0] == 0 or byte_data[0] == 1
    is_tcp = byte_data[0] == 1
    #7.12
    # ip0 = str(IPAddress(_big_endian_bytes2int(byte_data[1:5])))
    ip0 = str(IPv4Address(_big_endian_bytes2int(byte_data[1:5])))
    # print(ip0)
    port0 = _big_endian_bytes2int(byte_data[5:7])
    #7.12
    # ip1 = str(IPAddress(_big_endian_bytes2int(byte_data[7:11])))
    ip1 = str(IPv4Address(_big_endian_bytes2int(byte_data[7:11])))
    port1 = _big_endian_bytes2int(byte_data[11:13])
    return {
        # 'protocol': 'TCP' if is_tcp else 'UDP',
        'ip0': ip0,
        'port0': port0,
        'ip1': ip1,
        'port1': port1,
    }

def extract_1dimages(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)

    ### kc, changed magic number from 2051 to 2050
    if magic != 2050:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    leng = _read32(bytestream)
    print('num_images & leng of image:')
    print(num_images, leng)

    # shape = 2
    # dimensions = []
    # for _ in range(shape - 1):
    #   dimensions.append(int.from_bytes(.read(4), byteorder='big'))
    # print('dimensions', dimensions)
    sess_list = []
    data_list = []

    for _ in range(num_images):
      each_data_point = bytestream.read(leng)
      each_data_point = numpy.frombuffer(each_data_point, dtype=numpy.uint8)
      # print('each data point', numpy.array(each_data_point).shape)
      # print(each_data_point)
      session_info = _extract_session_info(each_data_point[:SESSION_BYTE_LEN])
      sess_list.append(session_info)
      other_data = each_data_point[SESSION_BYTE_LEN:]
      data_list.append(other_data)

    sess = numpy.array(sess_list)
    data = numpy.array(data_list)


    # buf = bytestream.read(leng * num_images)
    # data = numpy.frombuffer(buf, dtype=numpy.uint8)
    # print(data)
    # data = data.reshape(num_images, leng, 1)
    return sess, data

## kc, change num_classes from 2 to ..
def dense_to_one_hot(labels_dense, num_classes=4):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    print(num_items)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels


class DataSet(object):

  def __init__(self, five_tuple, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)

      ## kc, delete the images.shape[3] == 1
      # assert images.shape[3] == 1

      ## kc, change images.shape[1] * images.shape[2] to images.shape[1]

      images = images.reshape(images.shape[0],
                              images.shape[1])
      if dtype == tf.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._five_tuple = five_tuple
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def five_tuple(self):
    return self._five_tuple

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      s = "EPOCH: %d" % (self._epochs_completed)
      print(s)
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate(
          (images_rest_part, images_new_part), axis=0), numpy.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()

  id = int(train_dir[-1])
  if id == 0:
    id = 10

  if fake_data:
    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
    data_sets.train = fake()
    data_sets.validation = fake()
    data_sets.test = fake()
    return data_sets

  # TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  # TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  # TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  # TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  # kc
  # TRAIN_IMAGES = 'tcp-train-images-idx2-ubyte.gz'
  # TRAIN_LABELS = 'tcp-train-labels-idx1-ubyte.gz'
  # TEST_IMAGES = 'tcp-test-images-idx2-ubyte.gz'
  # TEST_LABELS = 'tcp-test-labels-idx1-ubyte.gz'

  ## kc, change to 5 pkt,
  # TRAIN_IMAGES = '5pkts-subflow-skype-train-images-idx2-ubyte.gz'
  # TRAIN_LABELS = '5pkts-subflow-skype-train-labels-idx1-ubyte.gz'
  # TEST_IMAGES = '5pkts-subflow-skype-test-images-idx2-ubyte.gz'
  # TEST_LABELS = '5pkts-subflow-skype-test-labels-idx1-ubyte.gz'

  # # kc, added 1pkt type
  TRAIN_IMAGES = str(id)+'pkts-subflow-skype-train-images-idx2-ubyte.gz'
  TRAIN_LABELS = str(id)+'pkts-subflow-skype-train-labels-idx1-ubyte.gz'
  TEST_IMAGES = str(id)+'pkts-subflow-skype-test-images-idx2-ubyte.gz'
  TEST_LABELS = str(id)+'pkts-subflow-skype-test-labels-idx1-ubyte.gz'

#  kc, comment validation dataset
#   VALIDATION_SIZE = 70

  local_file = maybe_download(TRAIN_IMAGES, train_dir)
  train_tuple, train_images = extract_1dimages(local_file)

  local_file = maybe_download(TRAIN_LABELS, train_dir)
  train_labels = extract_labels(local_file, one_hot=one_hot)

  local_file = maybe_download(TEST_IMAGES, train_dir)
  test_tuple, test_images = extract_1dimages(local_file)

  local_file = maybe_download(TEST_LABELS, train_dir)
  test_labels = extract_labels(local_file, one_hot=one_hot)
  #
  # validation_images = train_images[:VALIDATION_SIZE]
  # validation_labels = train_labels[:VALIDATION_SIZE]
  # train_images = train_images[VALIDATION_SIZE:]
  # train_labels = train_labels[VALIDATION_SIZE:]

  data_sets.train = DataSet(train_tuple, train_images, train_labels, dtype=dtype)
  # data_sets.validation = DataSet(validation_images, validation_labels,
  #                                dtype=dtype)
  data_sets.test = DataSet(test_tuple, test_images, test_labels, dtype=dtype)

  return data_sets