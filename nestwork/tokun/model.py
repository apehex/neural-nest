"""Piece together the actual VAE CNN model for tokun."""

import functools

import keras
import tensorflow as tf

import mlable.layers.shaping

# ENCODER #####################################################################

@keras.saving.register_keras_serializable(package='models')
class Encoder(tf.keras.models.Model):
    def __init__(
        self,
        token_dim: int,
        input_dim: int,
        sequence_axis: int=1,
        feature_axis: int=-1,
        **kwargs
    ) -> None:
        # init
        super(Encoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': token_dim,
            'input_dim': input_dim,
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,}
        # layers
        self._factor = tf.cast(1. / input_dim, tf.float32)
        self._divide = mlable.layers.shaping.Divide(axis=sequence_axis, factor=token_dim, insert=True, right=True, name='reshaping') # (B, S * T,) => (B, S, T)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return tf.cast(self._factor, inputs.dtype) * self._divide(inputs)

    def get_config(self) -> dict:
        __config = super(Encoder, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# DECODER #####################################################################

@keras.saving.register_keras_serializable(package='models')
class Decoder(tf.keras.models.Model):
    def __init__(
        self,
        token_dim: int,
        output_dim: int,
        sequence_axis: int=1,
        feature_axis: int=-1,
        **kwargs
    ) -> None:
        # init
        super(Decoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': token_dim,
            'output_dim': output_dim,
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,}
        # layers
        self._factor = tf.cast(output_dim, tf.float32)
        self._divide = mlable.layers.shaping.Divide(axis=feature_axis, factor=token_dim, insert=False, right=False, name='reshaping') # (B, S, T * E) => (B, S * T, E)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.cast(self._factor, inputs.dtype) * self._divide(inputs)

    def get_config(self) -> dict:
        __config = super(Decoder, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# VAE #########################################################################

@keras.saving.register_keras_serializable(package='models')
class AutoEncoder(tf.keras.models.Model):
    def __init__(
        self,
        token_dim: list,
        input_dim: int,
        output_dim: int,
        sequence_axis: int=1,
        feature_axis: int=-1,
        **kwargs
    ) -> None:
        # init
        super(AutoEncoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': token_dim,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,}
        # layers
        self._encoder = Encoder(token_dim=token_dim, input_dim=input_dim, sequence_axis=sequence_axis, feature_axis=feature_axis)
        self._decoder = Decoder(token_dim=token_dim, output_dim=output_dim, sequence_axis=sequence_axis, feature_axis=feature_axis)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._decoder(self._encoder(x))

    def get_config(self) -> dict:
        __config = super(AutoEncoder, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
