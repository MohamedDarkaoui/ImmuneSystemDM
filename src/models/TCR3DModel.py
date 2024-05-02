from src.models.AbstractModel import AbstractModel
from tensorflow import keras


class TCR3DModel(AbstractModel):
    """
    This model subclass is meant for training on single tcr chains, such as alpha or beta or a combination of both
    """

    def __init__(
            self,
            save_path,
            input_shape,
            kernel_size=3,
            l2=0.01,
            nr_filters_layer1=128,
            nr_filters_layer2=64,
            nr_dense=32,
            dense_activation='relu',
            optimizer='rmsprop',
            batch_size=32,
            pool_size=2,
            dropout_rate=0.25
    ):
        self.kernel_size = kernel_size
        self.l2 = l2
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        super().__init__(save_path, input_shape, nr_filters_layer1, nr_filters_layer2, nr_dense, dense_activation,
                         optimizer, batch_size)

    def max_pool(self):
        return keras.layers.MaxPool2D(pool_size=(self.pool_size, self.pool_size))

    def spacial_dropout(self):
        return keras.layers.SpatialDropout2D(self.dropout_rate)

    def create_conv(self, nr_filters: int, **kwargs):
        return keras.layers.Conv2D(
            nr_filters,
            (self.kernel_size, self.kernel_size),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(self.l2),
            **kwargs
        )
