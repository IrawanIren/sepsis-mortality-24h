"""Model definition and single-fold training.

Architecture follows the proposal: a CNN and bidirectional GRU pathway
for the hourly sequence, an MLP pathway for the static context, merged
before a sigmoid output. Input dimensions are read from the arrays, so
changes to the feature set require no edits here.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from metrics import compute_metrics
from windows import fit_transform_fold, make_windows

# Search space, unchanged from the proposal.
HP_SPACE = {
    'learning_rate': [1e-3, 1e-4, 5e-5],
    'l2':            [1e-2, 5e-3, 1e-3],
    'gru1':          [32, 48, 64],
    'gru2':          [16, 24, 32],
    'dropout':       [0.2, 0.3, 0.4],
}

DEFAULT_HP = {'learning_rate': 1e-3, 'l2': 1e-3,
              'gru1': 32, 'gru2': 16, 'dropout': 0.3}

# Ablation variants.
VARIANTS = {
    'full':     dict(cnn=True,  mlp=True),   # proposed
    'no_mlp':   dict(cnn=True,  mlp=False),  # no static pathway
    'no_cnn':   dict(cnn=False, mlp=True),   # no convolutional block
    'gru_only': dict(cnn=False, mlp=False),  # dynamic pathway alone
}


def build_model(window_size, n_dynamic, n_static, hp, variant='full'):
    """Build the network for one ablation variant."""
    spec = VARIANTS[variant]
    reg = keras.regularizers.l2(hp['l2'])

    dyn_in = layers.Input(shape=(window_size, n_dynamic), name='dynamic')
    x = dyn_in

    if spec['cnn']:
        # 'same' padding keeps short windows (W=6) usable after pooling.
        x = layers.Conv1D(64, 3, padding='same', activation='relu',
                          kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        if window_size >= 4:
            x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(hp['dropout'])(x)

    x = layers.Bidirectional(layers.GRU(
        hp['gru1'], return_sequences=True, kernel_regularizer=reg))(x)
    x = layers.Dropout(hp['dropout'])(x)
    x = layers.Bidirectional(layers.GRU(
        hp['gru2'], kernel_regularizer=reg))(x)

    inputs = [dyn_in]
    if spec['mlp']:
        stat_in = layers.Input(shape=(n_static,), name='static')
        s = layers.Dense(32, activation='relu', kernel_regularizer=reg)(stat_in)
        s = layers.BatchNormalization()(s)
        s = layers.Dropout(hp['dropout'])(s)
        s = layers.Dense(16, activation='relu', kernel_regularizer=reg)(s)
        x = layers.Concatenate()([x, s])
        inputs.append(stat_in)

    x = layers.Dense(32, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(hp['dropout'])(x)
    # float32 output keeps the sigmoid stable under mixed precision.
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = keras.Model(inputs, out)
    model.compile(
        optimizer=keras.optimizers.Adam(hp['learning_rate']),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc')],
    )
    return model


def train_fold(dyn, stat, y, train_idx, val_idx, window_size, step,
               hp, variant='full', seed=42, epochs=40, batch_size=512,
               n_hours=24, verbose=0):
    """Train on one fold.

    Returns
    -------
    metrics : dict          patient-level metrics on the validation fold
    y_prob  : (n_val,)      one probability per validation patient
    rows    : (n_val,)      row indices of those patients in `dyn`
    """
    keras.utils.set_random_seed(seed)

    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[train_idx] = True
    dyn_s, stat_s = fit_transform_fold(dyn, stat, train_mask)

    Xd_tr, Xs_tr, y_tr, _ = make_windows(
        dyn_s, stat_s, y, train_idx, window_size, step,
        training=True, n_hours=n_hours)
    Xd_va, Xs_va, y_va, rows = make_windows(
        dyn_s, stat_s, y, val_idx, window_size, step,
        training=False, n_hours=n_hours)

    use_static = VARIANTS[variant]['mlp']
    inp_tr = [Xd_tr, Xs_tr] if use_static else Xd_tr
    inp_va = [Xd_va, Xs_va] if use_static else Xd_va

    model = build_model(window_size, dyn.shape[2], stat.shape[1],
                        hp, variant)

    # Class weights counteract the 15% event rate. They improve ranking
    # but tend to push calibration slope below 1, which is reported
    # rather than corrected at this stage.
    pos = y_tr.sum()
    cw = {0: len(y_tr) / (2 * (len(y_tr) - pos)),
          1: len(y_tr) / (2 * pos)}

    model.fit(
        inp_tr, y_tr,
        validation_data=(inp_va, y_va),
        epochs=epochs, batch_size=batch_size, class_weight=cw,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor='val_auc', mode='max', patience=4,
            restore_best_weights=True)],
        verbose=verbose,
    )

    # Validation windows are right-aligned only, so predictions are
    # already one per patient.
    y_prob = model.predict(inp_va, batch_size=1024, verbose=0).ravel()
    metrics = compute_metrics(y_va, y_prob)

    keras.backend.clear_session()
    del model

    return metrics, y_prob, rows
