import tensorflow as tf


class Model:
    def __init__(self) -> None:
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(100, input_shape=(None, 102)),
                tf.keras.layers.Dense(1000, activation="relu"),
                tf.keras.layers.Dense(2000, activation="relu"),
                tf.keras.layers.Dense(1000, activation="relu"),
                tf.keras.layers.Dense(500, activation="relu"),
                tf.keras.layers.Dense(250, activation="relu"),
                tf.keras.layers.Dense(10, activation="relu"),
                tf.keras.layers.Dense(1, activation="linear"),
            ]
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(),
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=[
                tf.keras.metrics.mean_squared_error,
                tf.keras.metrics.mean_absolute_error,
            ],
        )
