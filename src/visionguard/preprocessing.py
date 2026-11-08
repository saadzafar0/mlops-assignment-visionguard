"""The unchanged training-time RGB and padded-resize transform, shared with serving."""

from visionguard.runtime import tf


def decode_and_resize(encoded, image_size: int):
    image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
    image = tf.ensure_shape(image, (None, None, 3))
    image = tf.image.resize_with_pad(image, image_size, image_size, antialias=True)
    return tf.clip_by_value(image, 0.0, 255.0)
