"""Comparator architectures evaluated under the same protocol.

Both models receive the identical input representation, window
configuration, data partition, and tuning budget as the proposed model,
so that differences reflect the sequence encoder rather than the setup.
The static pathway, the concatenation, and the output head are shared.

One structural difference is not controlled for and is stated in the
manuscript: the proposed model projects the raw channels through a
convolutional layer with batch normalisation before the recurrent
layers, whereas the bidirectional LSTM operates on raw channels
directly, as in the standard formulation.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


BILSTM_SPACE = {
    'learning_rate': [1e-3, 1e-4, 5e-5],
    'l2':            [1e-2, 5e-3, 1e-3],
    'lstm1':         [32, 48, 64],
    'lstm2':         [16, 24, 32],
    'dropout':       [0.2, 0.3, 0.4],
}

TRANSFORMER_SPACE = {
    'learning_rate': [1e-3, 1e-4, 5e-5],
    'l2':            [1e-2, 5e-3, 1e-3],
    'd_model':       [32, 64, 128],
    'n_heads':       [2, 4, 8],
    'ff_dim':        [64, 128, 256],
    'n_layers':      [1, 2, 3],
    'dropout':       [0.1, 0.2, 0.3],
}


def _static_pathway(stat_in, reg, dropout):
    """Identical to the proposed model, so the comparison isolates the
    sequence encoder."""
    s = layers.Dense(32, activation='relu', kernel_regularizer=reg)(stat_in)
    s = layers.BatchNormalization()(s)
    s = layers.Dropout(dropout)(s)
    return layers.Dense(16, activation='relu', kernel_regularizer=reg)(s)


def _head(x, s, reg, dropout):
    x = layers.Concatenate()([x, s])
    x = layers.Dense(32, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    return layers.Dense(1, activation='sigmoid', dtype='float32')(x)


def build_bilstm(window_size, n_dynamic, n_static, hp):
    """Two stacked bidirectional LSTM layers."""
    reg = keras.regularizers.l2(hp['l2'])
    dyn_in = layers.Input(shape=(window_size, n_dynamic), name='dynamic')
    stat_in = layers.Input(shape=(n_static,), name='static')

    x = layers.Bidirectional(layers.LSTM(
        hp['lstm1'], return_sequences=True, kernel_regularizer=reg))(dyn_in)
    x = layers.Dropout(hp['dropout'])(x)
    x = layers.Bidirectional(layers.LSTM(
        hp['lstm2'], kernel_regularizer=reg))(x)

    s = _static_pathway(stat_in, reg, hp['dropout'])
    out = _head(x, s, reg, hp['dropout'])

    model = keras.Model([dyn_in, stat_in], out)
    model.compile(optimizer=keras.optimizers.Adam(hp['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.AUC(name='auc')])
    return model


def build_transformer(window_size, n_dynamic, n_static, hp):
    """Transformer encoder with learned positional embeddings, followed
    by mean pooling over time.

    Sequences here are short and of fixed length, so learned embeddings
    are used rather than sinusoidal encoding.
    """
    reg = keras.regularizers.l2(hp['l2'])
    d_model, n_heads = hp['d_model'], hp['n_heads']

    dyn_in = layers.Input(shape=(window_size, n_dynamic), name='dynamic')
    stat_in = layers.Input(shape=(n_static,), name='static')

    # Attention requires a uniform channel dimension, so the raw
    # channels are projected first.
    x = layers.Dense(d_model, kernel_regularizer=reg)(dyn_in)
    pos = layers.Embedding(input_dim=window_size, output_dim=d_model)(
        tf.range(window_size))
    x = x + pos

    for _ in range(hp['n_layers']):
        attn = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=max(d_model // n_heads, 1),
            dropout=hp['dropout'])(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff = layers.Dense(hp['ff_dim'], activation='relu',
                          kernel_regularizer=reg)(x)
        ff = layers.Dropout(hp['dropout'])(ff)
        ff = layers.Dense(d_model, kernel_regularizer=reg)(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(hp['dropout'])(x)

    s = _static_pathway(stat_in, reg, hp['dropout'])
    out = _head(x, s, reg, hp['dropout'])

    model = keras.Model([dyn_in, stat_in], out)
    model.compile(optimizer=keras.optimizers.Adam(hp['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.AUC(name='auc')])
    return model


BUILDERS = {'BiLSTM': build_bilstm, 'Transformer': build_transformer}
SPACES = {'BiLSTM': BILSTM_SPACE, 'Transformer': TRANSFORMER_SPACE}
