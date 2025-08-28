"""Large Transformer builder.

This module creates a configurable transformer encoder model intended to
demonstrate an architecture that can be scaled toward billions of parameters.
It includes safety checks so attempting to build >3B parameters will warn the
user and require explicit confirmation via `force=True`.
"""
from __future__ import annotations

from typing import Optional
import tensorflow as tf


def _dense_relu_layer(units):
    return tf.keras.layers.Dense(units, activation='relu')


def build_large_transformer(seq_length: int = 128,
                            n_features: int = 16,
                            d_model: int = 1024,
                            n_heads: int = 16,
                            n_layers: int = 12,
                            mlp_dim: int = 4096,
                            output_dim: int = 1,
                            force: bool = False) -> tf.keras.Model:
    """Build a transformer encoder model.

    Parameters control model size. Rough parameter count = O(n_layers * d_model * mlp_dim).
    """
    # very rough parameter count estimate
    est_params = n_layers * (d_model * mlp_dim * 2 + d_model * d_model)
    if est_params > 3_200_000_000 and not force:
        raise ValueError(f"Estimated params ~{est_params:,} exceed 3.2B. Use force=True to proceed.")

    inputs = tf.keras.Input(shape=(seq_length, n_features), name='input_sequence')

    # project features to model dim
    x = tf.keras.layers.Dense(d_model, name='proj_in')(inputs)

    # positional encoding (learned)
    pos_emb = tf.keras.layers.Embedding(seq_length, d_model, name='pos_emb')(
        tf.range(start=0, limit=seq_length, delta=1))
    x = x + pos_emb

    for i in range(n_layers):
        # Layer norm + MultiHeadAttention
        ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'ln1_{i}')(x)
        attn = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads, name=f'attn_{i}')(ln1, ln1)
        x = tf.keras.layers.Add(name=f'residual_attn_{i}')([x, attn])

        ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'ln2_{i}')(x)
        mlp = tf.keras.layers.Dense(mlp_dim, activation='gelu', name=f'mlp_fc1_{i}')(ln2)
        mlp = tf.keras.layers.Dense(d_model, name=f'mlp_fc2_{i}')(mlp)
        x = tf.keras.layers.Add(name=f'residual_mlp_{i}')([x, mlp])

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='ln_final')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(output_dim, activation='linear', name='price_out')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='LargeTransformer')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
    return model


if __name__ == '__main__':
    # quick smoke test
    m = build_large_transformer(seq_length=64, n_features=8, d_model=256, n_heads=8, n_layers=2, mlp_dim=1024)
    m.summary()
