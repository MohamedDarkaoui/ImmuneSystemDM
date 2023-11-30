from tensorflow import keras


def create_conv(nr_filters: int, **kwargs):
    return keras.layers.Conv2D(
            nr_filters,
            (3, 3),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(0.01),
            **kwargs
        )


def add_unit(model, shape=None):
    if shape:
        model.add(create_conv(128, input_shape=shape))
    else:
        model.add(create_conv(128))
    model.add(keras.layers.BatchNormalization())
    model.add(create_conv(64))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.SpatialDropout2D(0.25))
    model.add(keras.layers.BatchNormalization())


def compile_model(model):
    model.compile(
        optimizer='rmsprop',
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            keras.metrics.AUC(curve="ROC", name="roc_auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ],
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None,
        distribute=None,
    )
    return model


def build_model(input_shape: tuple):
    model = keras.models.Sequential()
    # unit 1
    add_unit(model, input_shape)
    # unit 2
    add_unit(model)
    #  flatten
    model.add(keras.layers.Flatten())
    # FC layer
    model.add((keras.layers.Dense(32, 'relu')))
    # output layer
    model.add(keras.layers.Dense(1, 'sigmoid'))

    model = compile_model(model)
    return model


