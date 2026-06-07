"""
CNN-LSTM encoder + DQN agent with the class-frequency-aware reward, carried
over from the JISA paper (CNN-LSTM-DQN-IDS) and adapted for the coarse,
cross-dataset label space.

Reward (missing an attack of family c costs lambda_c = log(N / N_c)):
    correct decision        -> +alpha
    false negative (missed)  -> -beta * lambda_c   <- the imbalance correction
    false positive           -> -phi

This is the SAME mechanism as paper 1; the contribution of paper 2 is whether
this reward improves CROSS-dataset transfer, not the mechanism itself.

Requires tensorflow>=2.16. Kept dependency-light and CPU-runnable for a smoke
test; use a GPU build for full runs.
"""

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    _HAS_TF = True
except Exception:  # allow import on machines without TF for inspection
    _HAS_TF = False


def build_cnn_lstm_encoder(n_features: int, embed_dim: int = 128):
    """1D-CNN over the flow feature vector, then LSTM, then a dense embedding."""
    if not _HAS_TF:
        raise ImportError("tensorflow is required to build the model")
    inp = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dense(embed_dim, activation="relu", name="embedding")(x)
    return Model(inp, x, name="cnn_lstm_encoder")


def build_q_network(n_features: int, n_actions: int, embed_dim: int = 128):
    """Encoder + Q-head. Actions map to {allow, alert, block} by default."""
    if not _HAS_TF:
        raise ImportError("tensorflow is required to build the model")
    enc = build_cnn_lstm_encoder(n_features, embed_dim)
    q = layers.Dense(64, activation="relu")(enc.output)
    q = layers.Dense(n_actions, activation="linear", name="q_values")(q)
    return Model(enc.input, q, name="cnn_lstm_dqn")


def lambda_c_from_counts(class_counts: np.ndarray) -> np.ndarray:
    """lambda_c = log(N / N_c), computed from TRAIN counts only."""
    counts = np.asarray(class_counts, dtype=np.float64)
    N = counts.sum()
    return np.log(N / np.maximum(counts, 1.0))


def reward(action: int, true_class: int, lambda_c: np.ndarray,
           alpha: float = 5.0, beta: float = 2.0, phi: float = 3.0,
           normal_id: int = 0) -> float:
    """
    action: 0=allow, 1=alert, 2=block
    A missed attack (true is attack but action=allow) is penalised by
    beta * lambda_c[true_class]; a false positive (true=Normal, action=block)
    by phi; correct handling rewarded by alpha.
    """
    is_attack = true_class != normal_id
    if is_attack:
        if action == 0:                      # missed an attack: the costly error
            return -beta * float(lambda_c[true_class])
        return alpha                          # alerted/blocked an attack
    else:
        if action == 2:                       # blocked benign traffic
            return -phi
        return alpha                          # correctly allowed benign


# Default config (Table 2 carryover)
DEFAULT_CONFIG = dict(
    embed_dim=128, n_actions=3,
    lr=1e-3, gamma=0.99, buffer=10_000, batch=64,
    target_sync=500, eps_start=1.0, eps_end=0.1, eps_decay_episodes=1_000,
    huber_delta=1.0, alpha=5.0, beta=2.0, phi=3.0,
    seeds=(42, 7, 13, 99, 123),
)
