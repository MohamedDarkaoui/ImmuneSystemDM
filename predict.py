import pandas as pd
import tensorflow as tf
from src.data.data_processing import *

shape = (41, 13, 4)
model = tf.keras.models.load_model('models/binary_classification_concatenate')
model.summary()
df = pd.read_csv('data/test.csv')
labels = []
for index, row in df.iterrows():
    epitope = row['Peptide']
    TRA_CDR3 = row['CDR3a_extended']
    TRB_CDR3 = row['CDR3b_extended']

    imap = generate_padded_imaps(
        tcr_chains=[TRA_CDR3+TRB_CDR3],
        epitope=epitope,
        height=shape[0],
        width=shape[1]
    )[0]

    imap = np.expand_dims(imap, axis=0)
    prediction = model.predict(imap)
    assert len(prediction) == 1
    prediction = prediction[0]
    assert len(prediction) == 1
    prediction = prediction[0]
    threshold = 0.5
    class_label = (prediction >= threshold).astype(int)
    labels.append(class_label)

df['Prediction'] = labels
df.to_csv('out.csv', index=False)
