"""Pre and post processing pipelines for tokun."""

import functools
import itertools
import math

import tensorflow as tf

import mlable.maths.ops
import mlable.sampling
import mlable.shapes

# UNICODE #####################################################################

CODE_STX = b'\x02'
CODE_ETX = b'\x03'
CODE_FS = b'\x1c'
CODE_GS = b'\x1d'
CODE_RS = b'\x1e'
CODE_US = b'\x1f'

# ENCODE ######################################################################

def encode(data: tf.Tensor, token_size: int, sample_size: int, output_dtype: tf.dtypes.DType=tf.uint8) -> tf.Tensor:
    # factor 4 because of the UTF-32 encoding
    __dim = math.ceil(sample_size / token_size) * token_size
    # decode bytes from UTF-8
    __bytes = tf.strings.unicode_transcode(input=data, input_encoding='UTF-8', output_encoding='UTF-32-BE') # (B,)
    # decode byte strings to arrays of byte integers
    return tf.io.decode_raw(__bytes, out_type=output_dtype, fixed_length=__dim, little_endian=False) # (B, 4 * S) or (B, S) depending on the dtype (1 or 4 bytes)

# RESHAPE #####################################################################

def chunk(seq: list, size: int, repeats: bool=True) -> list:
    __chunks = (seq[__i:__i+size] for __i in range(0, len(seq), size))
    return list(__chunks if repeats else set(__chunks))

def merge(chunks: list) -> list:
    return list(itertools.chain.from_iterable(chunks))

def shape(expand: list=[]) -> list:
    return expand + [-1]

def reshape(data: tf.Tensor, expand: list=[]) -> tf.Tensor:
    # group by token unit
    __shape = shape(expand=expand)
    # partition or flatten the data
    return tf.reshape(tensor=data, shape=__shape) # for example (-1, G, G, G) the first dimension is not B

# AUGMENT #####################################################################

def offset(data: tf.Tensor, ticks: int=1) -> tf.Tensor:
    return tf.convert_to_tensor([ticks * b'\x00']) + data

# DECODE ######################################################################

def codepoint(data: tf.Tensor) -> tf.Tensor:
    # make sure the dtype is large enough for UTF-32 codepoints
    __data = tf.cast(data, dtype=tf.dtypes.int32)
    # group the bytes 4 by 4
    __shape = mlable.shapes.divide(shape=data.shape, axis=-1, factor=4, insert=True, right=True)
    __bytes = tf.reshape(tensor=__data, shape=__shape)
    # compute the UTF-32-BE codepoints
    return mlable.maths.ops.reduce_base(data=__bytes, base=256, axis=-1, keepdims=False)

def decode(data: tf.Tensor) -> tf.Tensor:
    # input = array of unicode codepoints
    __utf32 = tf.strings.unicode_encode(data, output_encoding='UTF-32-BE')
    # convert to standard UTF-8
    return tf.strings.unicode_transcode(input=__utf32, input_encoding='UTF-32-BE', output_encoding='UTF-8')

# > ###########################################################################

def preprocess(text: str, token_size: int, output_dtype: tf.dtypes.DType=tf.uint8, expand: list=[]) -> tf.Tensor:
    # as tensor
    __data = tf.convert_to_tensor(text, dtype=tf.dtypes.string)
    # list of bytes
    __bytes = encode(data=__data, token_size=token_size, sample_size=4 * len(text), output_dtype=output_dtype)
    # expand with unitary batch dim + cast
    return tf.cast(reshape(data=__bytes, expand=expand), tf.float32)

# < ###########################################################################

def unpad(text: str) -> str:
    return text.strip('\x00')

def unpack(data: tf.Tensor) -> list:
    __data = data.numpy().tolist()
    return [__s.decode('utf-8') for __s in __data]

def postprocess(prediction: tf.Tensor, feature: str='binary', value: str='bytes', random: bool=False) -> str:
    # interpret the feature axis
    if feature == 'binary':
        __output = mlable.sampling.binary(prediction=prediction, threshold=0.5, random=random)
    elif feature == 'categorical':
        __output = mlable.sampling.categorical(prediction=prediction, random=random)
    else:
        __output = tf.squeeze(tf.cast(tf.round(prediction), tf.int32), axis=-1)
    # merge the bytes into codepoints
    if value == 'bytes':
        __output = codepoint(data=__output)
    # decode the UTF-32-BE codepoints
    return decode(data=__output)

# SAMPLING ####################################################################

def sample(model: tf.keras.models.Model, text: str, **kwargs) -> tuple:
    __x = preprocess(text=text, token_size=kwargs.get('token_size', 16), expand=kwargs.get('expand', [1]), output_dtype=kwargs.get('output_dtype', tf.uint8))
    __e = model._encoder(__x)
    __p = model._decoder(__e)
    __y = postprocess(__p, binary=kwargs.get('binary', False), random=kwargs.get('random', False))
    __o = unpack(__y)
    return (__x, __e, __p, __y, __o)
