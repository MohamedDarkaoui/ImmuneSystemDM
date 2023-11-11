from tensorflow.keras.regularizers import l2

INTERACTION_MAP_CONFIG = {
  'features': 'hydrophob,isoelectric,mass,hydrophil',
  'operator': 'absdiff'
}

MODEL_CONFIG = {
  'filters': [(128, 64), (128, 64)],  # two units, each with two convolutional layers, the filter size of layer 1 us 128
  'kernel_size': (3, 3),
  'pool_size': (2, 2),
  'activation': 'relu',
  'padding': 'same',
  'kernel_initializer': 'he_normal',
  'kernel_regularizer': l2(0.01),
  'dropout_conv': 0.25,
  'dense_activation': 'relu',
  'dense_units': 32,
  'nr_classes': 1,
  'final_activation': 'sigmoid'
}
