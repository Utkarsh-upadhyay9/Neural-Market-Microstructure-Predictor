"""Distributed training harness for very large models.

This module provides a simple trainer that reads TFRecord shards produced by
`src.data.massive_collector.write_tfrecord_shards` and trains the `LargeTransformer`.
It supports tf.distribute strategies and streaming input pipelines to handle
large datasets without loading everything into memory.
"""
from __future__ import annotations

import os
from typing import List, Optional

import tensorflow as tf

from ..models.large_transformer import build_large_transformer


def _parse_example(example_proto):
    feature_description = {
        'symbol': tf.io.FixedLenFeature([], tf.string),
        'sequence': tf.io.VarLenFeature(tf.float32),
        'seq_length': tf.io.FixedLenFeature([], tf.int64),
        'n_features': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.float32)
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    seq_len = tf.cast(parsed['seq_length'], tf.int32)
    n_feat = tf.cast(parsed['n_features'], tf.int32)
    dense_seq = tf.sparse.to_dense(parsed['sequence'])
    seq = tf.reshape(dense_seq, (seq_len, n_feat))
    label = parsed['label']
    return seq, label


def make_dataset(tfrecord_files: List[str], batch_size: int = 32, shuffle_buffer: int = 1000, repeat: bool = True):
    ds = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP')
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def train(tfrecord_dir: str,
          seq_length: int = 128,
          n_features: int = 16,
          d_model: int = 1024,
          n_heads: int = 16,
          n_layers: int = 12,
          mlp_dim: int = 4096,
          epochs: int = 10,
          steps_per_epoch: int = 100,
          batch_size: int = 64,
          model_save: str = 'models/large_transformer.keras',
          force_build: bool = False):
    # find shards
    files = [os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith('.gz') or f.endswith('.tfrecord')]
    if not files:
        raise ValueError('No TFRecord files found in ' + tfrecord_dir)

    strategy = tf.distribute.MirroredStrategy()
    print(f'[INFO] Using strategy: {strategy.__class__.__name__} with {strategy.num_replicas_in_sync} replicas')

    with strategy.scope():
        model = build_large_transformer(seq_length=seq_length, n_features=n_features, d_model=d_model,
                                        n_heads=n_heads, n_layers=n_layers, mlp_dim=mlp_dim, force=force_build)

    dataset = make_dataset(files, batch_size=batch_size)

    # simple callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_save, save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3)
    ]

    model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
    model.save(model_save)
    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord-dir', default='data/massive')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()

    train(args.tfrecord_dir, epochs=args.epochs, steps_per_epoch=args.steps, batch_size=args.batch)
