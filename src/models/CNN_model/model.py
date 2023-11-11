from tensorflow import keras


def create_conv(unit_nr: int, conv_nr: int, config: dict, **kwargs):
    return keras.layers.Conv2D(
            config['filters'][unit_nr][conv_nr],
            config['kernel_size'],
            activation=config['activation'],
            padding=config['padding'],
            kernel_initializer=config['kernel_initializer'],
            kernel_regularizer=config['kernel_regularizer'],
            **kwargs
        )


def build_model(config: dict):
    model = keras.models.Sequential()

    # unit 1
    model.add(create_conv(0, 0, config, input_shape=(21, 10, 4)))
    model.add(keras.layers.BatchNormalization())
    model.add(create_conv(0, 1, config))
    model.add(keras.layers.MaxPool2D(pool_size=config['pool_size']))
    model.add(keras.layers.SpatialDropout2D(config['dropout_conv']))
    model.add(keras.layers.BatchNormalization())

    # unit 2
    model.add(create_conv(1, 0, config))
    model.add(keras.layers.BatchNormalization())
    model.add(create_conv(1, 1, config))
    model.add(keras.layers.MaxPool2D(pool_size=config['pool_size']))
    model.add(keras.layers.SpatialDropout2D(config['dropout_conv']))
    model.add(keras.layers.BatchNormalization())

    #  flatten
    model.add(keras.layers.Flatten())

    # FC layer
    model.add((keras.layers.Dense(config['dense_units'], config['dense_activation'])))

    # output layer
    model.add(keras.layers.Dense(config['nr_classes'], config['final_activation']))

    return model

