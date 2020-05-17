# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Common TF utilities."""

from __future__ import absolute_import, division, print_function

import math

import six
import tensorflow as tf


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def gelu(x):
    """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
    cdf = 0.5 * (
        1.0 + tf.tanh((math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))
    )
    return x * cdf


def swish(features):
    """Computes the Swish activation function.
  The tf.nn.swish operation uses a custom gradient to reduce memory usage.
  Since saving custom gradients in SavedModel is currently not supported, and
  one would not be able to use an exported TF-Hub module for fine-tuning, we
  provide this wrapper that can allow to select whether to use the native
  TensorFlow swish operation, or whether to use a customized operation that
  has uses default TensorFlow gradient computation.
  Args:
    features: A `Tensor` representing preactivation values.
  Returns:
    The activation value.
  """
    features = tf.convert_to_tensor(features)
    return features * tf.nn.sigmoid(features)


def pack_inputs(inputs):
    """Pack a list of `inputs` tensors to a tuple.
  Args:
    inputs: a list of tensors.
  Returns:
    a tuple of tensors. if any input is None, replace it with a special constant
    tensor.
  """
    inputs = tf.nest.flatten(inputs)
    outputs = []
    for x in inputs:
        if x is None:
            outputs.append(tf.constant(0, shape=[], dtype=tf.int32))
        else:
            outputs.append(x)
    return tuple(outputs)


def unpack_inputs(inputs):
    """unpack a tuple of `inputs` tensors to a tuple.
  Args:
    inputs: a list of tensors.
  Returns:
    a tuple of tensors. if any input is a special constant tensor, replace it
    with None.
  """
    inputs = tf.nest.flatten(inputs)
    outputs = []
    for x in inputs:
        if is_special_none_tensor(x):
            outputs.append(None)
        else:
            outputs.append(x)
    x = tuple(outputs)

    # To trick the very pointless 'unbalanced-tuple-unpacking' pylint check
    # from triggering.
    if len(x) == 1:
        return x[0]
    return tuple(outputs)


def is_special_none_tensor(tensor):
    """Checks if a tensor is a special None Tensor."""
    return tensor.shape.ndims == 0 and tensor.dtype == tf.int32


# TODO(hongkuny): consider moving custom string-map lookup to keras api.
def get_activation(activation_string):
    """
    Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.
    :param activation_string: String name of the activation function.
    :return: A Python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string, it will return `activation_string`.
    :raises: ValueError: The `activation_string` does not correspond to a known
                activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act in ["linear", "relu", "tanh"]:
        return tf.keras.activations.get(act)
    elif act == "gelu":
        return tf.keras.activations.get(gelu)
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.
  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.
  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.
  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.
  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        raise ValueError(
            "For the tensor `%s`, the actual tensor rank `%d` (shape = %s) is not "
            "equal to the expected tensor rank `%s`"
            % (name, actual_rank, str(tensor.shape), str(expected_rank))
        )
