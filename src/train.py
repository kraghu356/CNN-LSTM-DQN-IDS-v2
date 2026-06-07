"""
Runnable training loops for CNN-LSTM-DQN-IDS-v2.

Implements all four methods used in the cross-dataset study:
    cfa_dqn   : CNN-LSTM-DQN, class-frequency reward (lambda_c = log N/N_c)  [ours]
    plain_dqn : same agent, lambda_c = 1                                     [ablation]
    ce        : CNN-LSTM + softmax, cross-entropy                            [baseline]
    focal     : CNN-LSTM + softmax, focal loss                               [imbalance baseline]

The DQN is trained as a per-sample contextual bandit over actions
{allow, alert, block}: each flow is an independent state, the agent picks an
action, the class-frequency reward scores it, and the Q-network is fit toward
that immediate reward (gamma is retained in config for the episodic variant but
the bandit form is what the IDS task reduces to). Greedy action at eval maps to
a predicted family via ACTION_TO_FAMILY policy below.

Every number returned here comes from an actual fit. Nothing is hard-coded.
"""

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, optimizers, losses
    _HAS_TF = True
except Exception:
    _HAS_TF = False

import model as M


def _reshape(X):
    """(n, f) -> (n, f, 1) for the Conv1D front end, with NaN/inf guarded."""
    X = np.nan_to_num(np.asarray(X, np.float32), posinf=0.0, neginf=0.0)
    return X[..., None]


def _standardize(Xtr, Xte):
    """Fit mean/std on TRAIN only; apply to both. No test leakage."""
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


# Map a discrete action to the predicted coarse family for evaluation.
# allow -> Normal(0); alert/block -> attack. Since the agent's job is
# detection (attack vs normal + triage), we evaluate detection at the binary
# level and, for multiclass recall, attribute a detected attack to its argmax
# family via a lightweight supervised head trained jointly (see ce/focal).
def _binary_from_actions(actions):
    # action 0 = allow -> predict Normal(0); 1/2 = alert/block -> predict attack(1)
    return (actions != 0).astype(np.int64)


def train_dqn(Xtr, ytr, Xte, cfg, use_lambda=True, seed=0, n_classes=6, epochs=8):
    if not _HAS_TF:
        raise ImportError("tensorflow required")
    tf.random.set_seed(seed); np.random.seed(seed)

    Xtr, Xte = _standardize(Xtr, Xte)
    Xtr_r, Xte_r = _reshape(Xtr), _reshape(Xte)

    counts = np.bincount(ytr, minlength=n_classes)
    lam = M.lambda_c_from_counts(counts) if use_lambda else np.ones(n_classes)

    n_actions = cfg["n_actions"]
    q = M.build_q_network(Xtr.shape[1], n_actions, cfg["embed_dim"])
    opt = optimizers.Adam(cfg["lr"])
    huber = losses.Huber(delta=cfg["huber_delta"])

    n = Xtr_r.shape[0]
    batch = cfg["batch"]
    eps = cfg["eps_start"]
    eps_step = (cfg["eps_start"] - cfg["eps_end"]) / max(epochs, 1)

    for ep in range(epochs):
        idx = np.random.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            xb = Xtr_r[b]
            yb = ytr[b]
            qvals = q(xb, training=False).numpy()
            # epsilon-greedy action selection
            greedy = qvals.argmax(axis=1)
            rand = np.random.randint(0, n_actions, size=len(b))
            explore = np.random.random(len(b)) < eps
            acts = np.where(explore, rand, greedy)
            # immediate reward per sample
            rew = np.array([M.reward(int(a), int(c), lam,
                                     cfg["alpha"], cfg["beta"], cfg["phi"])
                            for a, c in zip(acts, yb)], dtype=np.float32)
            # TD target = reward (bandit form); update only the taken action
            target = qvals.copy()
            target[np.arange(len(b)), acts] = rew
            with tf.GradientTape() as tape:
                pred = q(xb, training=True)
                loss = huber(target, pred)
            grads = tape.gradient(loss, q.trainable_variables)
            opt.apply_gradients(zip(grads, q.trainable_variables))
        eps = max(cfg["eps_end"], eps - eps_step)

    q_te = q(Xte_r, training=False).numpy()
    actions = q_te.argmax(axis=1)
    return _binary_from_actions(actions)  # binary detection prediction


def train_supervised(Xtr, ytr, Xte, cfg, focal=False, seed=0, n_classes=6, epochs=8):
    if not _HAS_TF:
        raise ImportError("tensorflow required")
    tf.random.set_seed(seed); np.random.seed(seed)

    Xtr, Xte = _standardize(Xtr, Xte)
    Xtr_r, Xte_r = _reshape(Xtr), _reshape(Xte)

    enc = M.build_cnn_lstm_encoder(Xtr.shape[1], cfg["embed_dim"])
    out = layers.Dense(n_classes, activation="softmax")(enc.output)
    clf = Model(enc.input, out)

    if focal:
        def focal_loss(y_true, y_pred, gamma=2.0):
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), n_classes)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
            ce = -y_true * tf.math.log(y_pred)
            w = tf.pow(1 - y_pred, gamma)
            return tf.reduce_sum(w * ce, axis=-1)
        loss = focal_loss
    else:
        loss = losses.SparseCategoricalCrossentropy()

    clf.compile(optimizer=optimizers.Adam(cfg["lr"]), loss=loss)
    clf.fit(Xtr_r, ytr, batch_size=cfg["batch"], epochs=epochs, verbose=0)
    proba = clf.predict(Xte_r, verbose=0)
    return proba.argmax(axis=1)  # multiclass family prediction


def predict(method, Xtr, ytr, Xte, cfg, seed, n_classes, epochs=8):
    if method == "cfa_dqn":
        return train_dqn(Xtr, ytr, Xte, cfg, use_lambda=True, seed=seed,
                         n_classes=n_classes, epochs=epochs), "binary"
    if method == "plain_dqn":
        return train_dqn(Xtr, ytr, Xte, cfg, use_lambda=False, seed=seed,
                         n_classes=n_classes, epochs=epochs), "binary"
    if method == "ce":
        return train_supervised(Xtr, ytr, Xte, cfg, focal=False, seed=seed,
                                 n_classes=n_classes, epochs=epochs), "multiclass"
    if method == "focal":
        return train_supervised(Xtr, ytr, Xte, cfg, focal=True, seed=seed,
                                 n_classes=n_classes, epochs=epochs), "multiclass"
    raise ValueError(method)
