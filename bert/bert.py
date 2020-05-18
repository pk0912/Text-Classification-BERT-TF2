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
"""The main BERT model and related functions."""

import copy
import json
import math
import six
import tensorflow as tf
from . import helpers


class BertConfig(object):
    """Configuration for `BertLayer`."""

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
    ):
        """Constructs BertConfig.
        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertLayer(tf.keras.layers.Layer):
    """BERT model ("Bidirectional Encoder Representations from Transformers").
      Example usage:
      ```python
      # Already been converted into WordPiece token ids
      input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
      input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
      token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
      config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)
      pooled_output = bert.BertLayer(config=config)(
        input_word_ids=input_word_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids)
      ...
      ```
      """

    def __init__(self, config, is_training=True):
        """
        Constructor for BertLayer.
        :param config: `BertConfig` instance.
        :param is_training: bool. true for training model, false for eval model. Controls
                whether dropout will be applied.
        """
        super(BertLayer, self).__init__()
        self.config = copy.deepcopy(config)
        if not is_training:
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_probs_dropout_prob = 0.0
        self.is_training = is_training
        self.embedding_lookup = None
        self.embedding_postprocessor = None
        self.encoder = None
        self.pooler = None

    def build(self, input_shape):
        """
        Defines layers needed for Bert Architecture
        :param input_shape: Shape of inputs passed to the __call__ method
        """
        self.embedding_lookup = tf.keras.layers.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            embeddings_initializer=helpers.create_initializer(
                self.config.initializer_range
            ),
            name="word_embeddings",
        )
        self.embedding_postprocessor = EmbeddingPostProcessor(
            use_token_type=True,
            token_type_vocab_size=self.config.type_vocab_size,
            use_position_embeddings=True,
            initializer_range=self.config.initializer_range,
            max_position_embeddings=self.config.max_position_embeddings,
            dropout_prob=self.config.hidden_dropout_prob,
            is_training=self.is_training,
            name="embedding_postprocessor",
        )
        self.encoder = Transformer(
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            intermediate_act_fn=self.config.hidden_act,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            initializer_range=self.config.initializer_range,
            is_training=self.is_training,
            do_return_all_layers=True,
            name="encoder",
        )
        self.pooler = tf.keras.layers.Dense(
            self.config.hidden_size,
            activation=helpers.get_activation("tanh"),
            kernel_initializer=helpers.create_initializer(
                self.config.initializer_range
            ),
            name="pooler",
        )

    def __call__(self, input_ids, input_mask=None, token_type_ids=None, **kwargs):
        """
        __call__ method for BertLayer
        :param input_ids: int32 Tensor of shape [batch_size, seq_length].
        :param input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
        :param token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        :param kwargs: keyword arguments.
        :return: Output tensor of the pooler layer of shape ([batch_size, hidden_size])
        """
        if input_mask is None:
            input_mask = tf.constant([0])
        if token_type_ids is None:
            token_type_ids = tf.constant([0])
        inputs = [input_ids, input_mask, token_type_ids]
        return super(BertLayer, self).__call__(inputs, **kwargs)

    def call(self, inputs, **kwargs):
        """
        Call method for bert layer
        :param inputs: contains input_ids, input_mask, token_type_ids
        :return: Output tensor of the pooler layer of shape ([batch_size, hidden_size])
        """
        input_ids = inputs[0]
        input_mask = inputs[1]
        token_type_ids = inputs[2]
        if len(input_mask.get_shape()) == 1:
            input_mask = None
        if len(token_type_ids.get_shape()) == 1:
            token_type_ids = None
        input_shape = helpers.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
        # Perform embedding lookup on the word ids.
        word_embeddings = self.embedding_lookup(input_ids)
        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        embedding_output = self.embedding_postprocessor(word_embeddings, token_type_ids)
        attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)
        all_encoder_layers = self.encoder(embedding_output, attention_mask, batch_size)
        sequence_output = all_encoder_layers[-1]
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        pooled_output = self.pooler(first_token_tensor)
        return pooled_output

    def get_config(self):
        config = super().get_config().copy()
        config.update(self.config.to_dict())
        return config


class EmbeddingPostProcessor(tf.keras.layers.Layer):
    """Performs various post-processing on a word embedding tensor."""

    def __init__(
        self,
        use_token_type=False,
        token_type_vocab_size=2,
        use_position_embeddings=True,
        initializer_range=0.02,
        max_position_embeddings=512,
        dropout_prob=0.1,
        is_training=True,
        **kwargs,
    ):
        """
        Constructor for EmbeddingPostProcessor Layer
        :param use_token_type: bool. Whether to add embeddings for `token_type_ids`.
        :param token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
        :param use_position_embeddings: bool. Whether to add position embeddings for the
            position of each token in the sequence.
        :param initializer_range:
        :param max_position_embeddings:
        :param dropout_prob: float. Dropout probability applied to the final output tensor.
        """
        super(EmbeddingPostProcessor, self).__init__()
        self.use_token_type = use_token_type
        self.token_type_vocab_size = token_type_vocab_size
        self.use_position_embeddings = use_position_embeddings
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.dropout_prob = dropout_prob
        self.initializer = helpers.create_initializer(self.initializer_range)
        self.is_training = is_training
        self.token_type_table = None
        self.full_position_embeddings = None
        self.layer_norm = None
        self.layer_dropout = None

    def build(self, input_shape):
        """
        Defines layers needed for EmbeddingPostProcessor
        :param input_shape: Shape of inputs passed to the __call__ method
        """
        input_tensor_shape, token_type_ids_shape = input_shape
        seq_length = input_tensor_shape[1]
        width = input_tensor_shape[2]
        if self.use_token_type:
            if token_type_ids_shape[-1] == 1:
                raise ValueError(
                    "`token_type_ids` must be specified if" "`use_token_type` is True."
                )
            self.token_type_table = self.add_weight(
                "token_type_embeddings",
                shape=[self.token_type_vocab_size, width],
                initializer=helpers.create_initializer(self.initializer_range),
            )
        if self.use_position_embeddings:
            tf.debugging.assert_less_equal(seq_length, self.max_position_embeddings)
            self.full_position_embeddings = self.add_weight(
                "position_embeddings",
                shape=[self.max_position_embeddings, width],
                initializer=helpers.create_initializer(self.initializer_range),
            )
        self.layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-3, name="layer_norm"
        )
        self.layer_dropout = tf.keras.layers.Dropout(rate=self.dropout_prob)

    def __call__(self, input_tensor, token_type_ids=None, **kwargs):
        """
        _call__ method for EmbeddingPostProcessor layer
        :param input_tensor: float Tensor of shape [batch_size, seq_length, embedding_size].
        :param token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            Must be specified if `use_token_type` is True.
        :param kwargs: keyword arguments
        :return: float tensor with same shape as `input_tensor`.
        """
        if token_type_ids is None:
            token_type_ids = tf.constant([0])
        inputs = [input_tensor, token_type_ids]
        return super(EmbeddingPostProcessor, self).__call__(inputs, **kwargs)

    def call(self, inputs, **kwargs):
        """
        call method for EmbeddingPostProcessor Layer
        :param inputs: contains input_tensor and token_type_ids
        :param kwargs: keyword arguments
        :return: float tensor with same shape as `input_tensor`.
        """
        input_tensor, token_type_ids = inputs
        input_shape = helpers.get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        output = input_tensor
        if self.use_token_type:
            # This vocab will be small so we always do one-hot here, since it is always
            # faster for a small vocabulary.
            flat_token_type_ids = tf.reshape(token_type_ids, [-1])
            one_hot_ids = tf.one_hot(
                flat_token_type_ids, depth=self.token_type_vocab_size
            )
            token_type_embeddings = tf.matmul(one_hot_ids, self.token_type_table)
            token_type_embeddings = tf.reshape(
                token_type_embeddings, [batch_size, seq_length, width]
            )
            output += token_type_embeddings

        if self.use_position_embeddings:
            position_embeddings = tf.slice(
                self.full_position_embeddings, [0, 0], [seq_length, -1]
            )
            position_embeddings = tf.expand_dims(position_embeddings, axis=0)
            output += position_embeddings
        output = self.layer_norm(output)
        output = self.layer_dropout(output, training=self.is_training)
        return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.
    Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = helpers.get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = helpers.get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError(
            "Input tensor must have at least rank 2. Shape = %s" % input_tensor.shape
        )
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = helpers.get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def transpose_for_scores(
    input_tensor, batch_size, num_attention_heads, seq_length, width
):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width]
    )

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor


class Transformer(tf.keras.layers.Layer):
    """
        Multi-headed, multi-layer Transformer from "Attention is All You Need".
        This is almost an exact implementation of the original Transformer encoder.
        See the original paper:
        https://arxiv.org/abs/1706.03762
        Also see:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        intermediate_act_fn="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        is_training=True,
        do_return_all_layers=False,
        **kwargs,
    ):
        """
        Constructor for Transformer Class
        :param hidden_size: int. Hidden size of the Transformer.
        :param num_attention_heads: int. Number of attention heads in the Transformer.
        :param intermediate_size: int. The size of the "intermediate" (a.k.a., feed forward) layer.
        :param intermediate_act_fn: function. The non-linear activation function to apply
            to the output of the intermediate/feed-forward layer.
        :param hidden_dropout_prob: float. Dropout probability for the hidden layers.
        :param attention_probs_dropout_prob: float. Dropout probability of the attention probabilities.
        :param initializer_range: float. Range of the initializer (stddev of truncated normal).
        :param is_training: If true, dropout will be done, else not.
        """
        super(Transformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_act_fn = intermediate_act_fn
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.seq_length = None
        self.input_width = None
        self.do_return_all_layers = do_return_all_layers
        self.input_tensor_shape = []
        self.layers = []

    def build(self, input_shape):
        """
        Initializes n transformer layers.
        :param input_shape: Shape of inputs passed to the __call__ method
        """
        self.input_tensor_shape = input_shape[0]
        self.seq_length = self.input_tensor_shape[1]
        self.input_width = self.input_tensor_shape[2]
        for i in range(self.num_hidden_layers):
            self.layers.append(
                TransformerLayer(
                    hidden_size=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    intermediate_size=self.intermediate_size,
                    intermediate_act_fn=self.intermediate_act_fn,
                    hidden_dropout_prob=self.hidden_dropout_prob,
                    attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                    initializer_range=self.initializer_range,
                    is_training=self.is_training,
                    seq_length=self.seq_length,
                    input_width=self.input_width,
                )
            )
        super(Transformer, self).build(input_shape)

    def __call__(self, input_tensor, attention_mask=None, batch_size=None, **kwargs):
        """
        __call__ method for Transformer class.
        :param input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
        :param attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length, seq_length],
            with 1 for positions that can be attended to and 0 in positions that should not be.
        :param kwargs: keyword arguments
        :return: float Tensor of shape [batch_size, seq_length, hidden_size], the final
            hidden layer of the Transformer.
        """
        if attention_mask is None:
            attention_mask = tf.constant([0])
        if batch_size is None:
            batch_size = tf.constant([0])
        return super(Transformer, self).__call__(
            [input_tensor, attention_mask, batch_size], **kwargs
        )

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
        input_tensor = inputs[0]
        attention_mask = inputs[1]
        batch_size = inputs[2]
        if len(attention_mask.get_shape()) == 1:
            attention_mask = None
        if len(batch_size.get_shape()) == 1:
            batch_size = None
        prev_output = reshape_to_matrix(input_tensor)
        all_layer_outputs = []
        for layer in self.layers:
            layer_input = prev_output
            prev_output = layer(layer_input, attention_mask, batch_size)
            all_layer_outputs.append(prev_output)

        if self.do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = reshape_from_matrix(
                    layer_output, [batch_size, self.seq_length, self.input_width]
                )
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = reshape_from_matrix(
                prev_output, [batch_size, self.seq_length, self.input_width]
            )
            return final_output


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        intermediate_act_fn="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        is_training=True,
        seq_length=None,
        input_width=None,
    ):
        """
        Constructor for Transformer Layer
        :param hidden_size: int. Hidden size of the Transformer.
        :param num_attention_heads: int. Number of attention heads in the Transformer.
        :param intermediate_size: int. The size of the "intermediate" (a.k.a., feed forward) layer.
        :param intermediate_act_fn: function. The non-linear activation function to apply
            to the output of the intermediate/feed-forward layer.
        :param hidden_dropout_prob: float. Dropout probability for the hidden layers.
        :param attention_probs_dropout_prob: float. Dropout probability of the attention probabilities.
        :param initializer_range: float. Range of the initializer (stddev of truncated normal).
        :param is_training: If true, dropout will be done, else not.
        """
        super(TransformerLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_act_fn = intermediate_act_fn
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.seq_length = seq_length
        self.input_width = input_width
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.attention_head = None
        self.attention_dense = None
        self.attention_dropout = None
        self.attention_norm = None
        self.intermediate_dense = None
        self.output_dense = None
        self.output_dropout = None
        self.output_norm = None

    def build(self, input_shape):
        """
        Defines layers needed for Transformer Architecture
        :param input_shape: Shape of inputs passed to the __call__ method
        """
        # The Transformer performs sum residuals on all layers so the input needs
        # to be the same as the hidden size.
        if self.input_width != self.hidden_size:
            raise ValueError(
                "The width of the input tensor (%d) != hidden size (%d)"
                % (self.input_width, self.hidden_size)
            )
        self.attention_head = AttentionLayer(
            num_attention_heads=self.num_attention_heads,
            size_per_head=self.attention_head_size,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            do_return_2d_tensor=True,
            from_seq_length=self.seq_length,
            to_seq_length=self.seq_length,
            is_training=self.is_training,
        )
        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        self.attention_dense = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=helpers.create_initializer(self.initializer_range),
        )
        self.attention_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
        self.attention_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-3, name="attention_layer_norm"
        )
        # The activation is only applied to the "intermediate" hidden layer.
        self.intermediate_dense = tf.keras.layers.Dense(
            self.intermediate_size,
            activation=helpers.get_activation(self.intermediate_act_fn),
            kernel_initializer=helpers.create_initializer(self.initializer_range),
        )
        # Down-project back to `hidden_size` then add the residual.
        self.output_dense = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=helpers.create_initializer(self.initializer_range),
        )
        self.output_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
        self.output_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-3, name="output_layer_norm"
        )
        super(TransformerLayer, self).build(input_shape)

    def __call__(self, input_tensor, attention_mask=None, batch_size=None, **kwargs):
        """
        __call__ method for TransformerLayer
        :param input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
        :param attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length, seq_length],
            with 1 for positions that can be attended to and 0 in positions that should not be.
        :param kwargs: keyword arguments
        :return: float Tensor of shape [batch_size, seq_length, hidden_size], the final
            hidden layer of the Transformer.
        """
        return super(TransformerLayer, self).__call__(
            [input_tensor, attention_mask, batch_size], **kwargs
        )

    def call(self, inputs, **kwargs):
        """
        call method for TransformerLayer
        :param inputs: contains input_tensor and attention_mask.
        :param kwargs: keyword arguments
        :return: float Tensor of shape [batch_size, seq_length, hidden_size], the final
            hidden layer of the Transformer.
        """
        input_tensor = inputs[0]
        attention_mask = inputs[1]
        batch_size = inputs[2]
        attention_heads = []
        layer_input = reshape_to_matrix(input_tensor)
        attention_head = self.attention_head(
            from_tensor=layer_input,
            to_tensor=layer_input,
            attention_mask=attention_mask,
            batch_size=batch_size,
        )
        attention_heads.append(attention_head)
        if len(attention_heads) == 1:
            attention_output = attention_heads[0]
        else:
            # In the case where we have other sequences, we just concatenate
            # them to the self-attention head before the projection.
            attention_output = tf.concat(attention_heads, axis=-1)
        attention_output = self.attention_dense(attention_output)
        attention_output = self.attention_dropout(
            attention_output, training=self.is_training
        )
        attention_output = self.attention_norm(attention_output + layer_input)
        intermediate_output = self.intermediate_dense(attention_output)
        layer_output = self.output_dense(intermediate_output)
        layer_output = self.output_dropout(layer_output, training=self.is_training)
        layer_output = self.output_norm(layer_output + attention_output)
        return layer_output


class AttentionLayer(tf.keras.layers.Layer):
    """
    Performs multi-headed attention from `from_tensor` to `to_tensor`.
    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.
    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].
    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.
    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.
    """

    def __init__(
        self,
        num_attention_heads=12,
        size_per_head=64,
        query_act=None,
        key_act=None,
        value_act=None,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        do_return_2d_tensor=False,
        from_seq_length=None,
        to_seq_length=None,
        is_training=True,
    ):
        """
        Constructor for Attention Layer
        :param num_attention_heads: int. Number of attention heads.
        :param size_per_head: int. Size of each attention head.
        :param attention_probs_dropout_prob: (optional) float. Dropout probability of the
            attention probabilities.
        :param initializer_range: float. Range of the weight initializer.
        :param do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
            * from_seq_length, num_attention_heads * size_per_head]. If False, the
            output will be of shape [batch_size, from_seq_length, num_attention_heads
            * size_per_head].
        :param batch_size: (Optional) int. If the input is 2D, this might be the batch size
            of the 3D version of the `from_tensor` and `to_tensor`.
        :param from_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `from_tensor`.
        :param to_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `to_tensor`.
        :param is_training: If true, dropout will be done, else not.
        """
        super(AttentionLayer, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor
        self.is_training = is_training
        self.from_seq_length = from_seq_length
        self.to_seq_length = to_seq_length
        self.batch_size = None

        # `query_layer` = [B*F, N*H]
        self.query_layer = tf.keras.layers.Dense(
            self.num_attention_heads * self.size_per_head,
            activation=query_act,
            kernel_initializer=helpers.create_initializer(self.initializer_range),
            name="query",
        )

        # `key_layer` = [B*T, N*H]
        self.key_layer = tf.keras.layers.Dense(
            self.num_attention_heads * self.size_per_head,
            activation=key_act,
            kernel_initializer=helpers.create_initializer(self.initializer_range),
            name="key",
        )

        # `value_layer` = [B*T, N*H]
        self.value_layer = tf.keras.layers.Dense(
            self.num_attention_heads * self.size_per_head,
            activation=value_act,
            kernel_initializer=helpers.create_initializer(self.initializer_range),
            name="value",
        )
        self.attention_probs_dropout = tf.keras.layers.Dropout(
            rate=self.attention_probs_dropout_prob
        )

    def __call__(
        self, from_tensor, to_tensor, attention_mask=None, batch_size=None, **kwargs
    ):
        """
        __call__ method for AttentionLayer
        :param from_tensor: float Tensor of shape [batch_size, from_seq_length,
            from_width].
        :param to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        :param attention_mask: (optional) int32 Tensor of shape [batch_size,
            from_seq_length, to_seq_length]. The values should be 1 or 0. The
            attention scores will effectively be set to -infinity for any positions in
            the mask that are 0, and will be unchanged for positions that are 1.
        :param kwargs: keyword arguments.
        :return: float Tensor of shape [batch_size, from_seq_length,
            num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
            true, this will be of shape [batch_size * from_seq_length,
            num_attention_heads * size_per_head]).
        """
        inputs = [from_tensor, to_tensor, attention_mask, batch_size]
        return super(AttentionLayer, self).__call__(inputs, **kwargs)

    def call(self, inputs, **kwargs):
        """
        call method for AttentionLayer
        :param inputs: contains from_tensor, to_tensor and attention_mask
        :param kwargs: keyword arguments.
        :return: float Tensor of shape [batch_size, from_seq_length,
            num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
            true, this will be of shape [batch_size * from_seq_length,
            num_attention_heads * size_per_head]).
        """
        from_tensor = inputs[0]
        to_tensor = inputs[1]
        attention_mask = inputs[2]
        self.batch_size = inputs[3]

        from_shape = helpers.get_shape_list(from_tensor, expected_rank=[2, 3])
        to_shape = helpers.get_shape_list(to_tensor, expected_rank=[2, 3])

        if len(from_shape) != len(to_shape):
            raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`."
            )

        if len(from_shape) == 3:
            self.batch_size = from_shape[0]
            self.from_seq_length = from_shape[1]
            self.to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            if (
                self.batch_size is None
                or self.from_seq_length is None
                or self.to_seq_length is None
            ):
                raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified."
                )

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(to_tensor)

        query_layer = self.query_layer(from_tensor_2d)
        key_layer = self.key_layer(to_tensor_2d)
        value_layer = self.value_layer(to_tensor_2d)

        # `query_layer` = [B, N, F, H]
        query_layer = transpose_for_scores(
            query_layer,
            self.batch_size,
            self.num_attention_heads,
            self.from_seq_length,
            self.size_per_head,
        )

        # `key_layer` = [B, N, T, H]
        key_layer = transpose_for_scores(
            key_layer,
            self.batch_size,
            self.num_attention_heads,
            self.to_seq_length,
            self.size_per_head,
        )

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(
            attention_scores, 1.0 / math.sqrt(float(self.size_per_head))
        )

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

            # Normalize the attention scores to probabilities.
            # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_probs_dropout(
            attention_probs, training=self.is_training
        )

        # `value_layer` = [B, T, N, H]
        value_layer = tf.reshape(
            value_layer,
            [
                self.batch_size,
                self.to_seq_length,
                self.num_attention_heads,
                self.size_per_head,
            ],
        )

        # `value_layer` = [B, N, T, H]
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)

        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if self.do_return_2d_tensor:
            # `context_layer` = [B*F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [
                    self.batch_size * self.from_seq_length,
                    self.num_attention_heads * self.size_per_head,
                ],
            )
        else:
            # `context_layer` = [B, F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [
                    self.batch_size,
                    self.from_seq_length,
                    self.num_attention_heads * self.size_per_head,
                ],
            )

        return context_layer
