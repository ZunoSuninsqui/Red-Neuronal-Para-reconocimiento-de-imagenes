from __future__ import annotations

from typing import Dict, Tuple

from tensorflow import keras
from tensorflow.keras import layers

from AccesoADatos import DIGIT_CLASS_NAMES


def build_canvas_digit_model(
    input_shape: Tuple[int, int, int],
    *,
    base_filters: int,
    dropout: float,
    weight_decay: float,
    augmentation_params: Dict[str, float],
) -> keras.Model:
    """Construct the convolutional architecture used for canvas digits."""

    regularizer = keras.regularizers.l2(weight_decay)

    data_augmentation = keras.Sequential(
        [
            layers.RandomRotation(augmentation_params.get("rotation", 0.1)),
            layers.RandomTranslation(
                augmentation_params.get("translation", 0.1),
                augmentation_params.get("translation", 0.1),
            ),
            layers.RandomZoom(augmentation_params.get("zoom", 0.1)),
            layers.RandomShear(augmentation_params.get("shear", 0.05)),
        ],
        name="data_augmentation",
    )

    def conv_block(x: keras.Tensor, filters: int) -> keras.Tensor:
        x = layers.Conv2D(
            filters,
            (3, 3),
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(
            filters,
            (3, 3),
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(dropout)(x)
        return x

    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Conv2D(
        base_filters,
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizer,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    filters = base_filters
    for _ in range(3):
        filters *= 2
        x = conv_block(x, filters)

    x = layers.Conv2D(filters * 2, (3, 3), padding="same", use_bias=False, kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizer)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(len(DIGIT_CLASS_NAMES))(x)

    model = keras.Model(inputs, outputs, name="canvas_digit_cnn")
    return model
