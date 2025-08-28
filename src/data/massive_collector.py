"""massive_collector

Utilities to generate very large synthetic datasets and write them as sharded TFRecord
files. The generator is configurable so you can create datasets that mimic market
microstructure (multi-feature timeseries) at massive scale. By default it is safe
and small; change parameters to scale up. The module writes sharded TFRecords and
supports streaming generation to avoid keeping all data in memory.

Note: Creating very large datasets (many GB/TB) will consume disk and time. Use
this responsibly and prefer cloud storage for truly massive datasets.
"""
from __future__ import annotations

import os
import math
import json
import gzip
import struct
import random
from typing import Iterable, Optional

import numpy as np
import tensorflow as tf


def _float_feature(value: float):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_list_feature(values: Iterable[float]):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))


def synthetic_bar_sequence(length: int, n_features: int, base_price: float = 100.0, seed: Optional[int] = None):
    """Generate a single synthetic multivariate time-series resembling price bars.

    Returns a numpy array of shape (length, n_features).
    """
    rs = np.random.RandomState(seed)
    # simple geometric Brownian-like walk with feature noise
    dt = 1.0
    mu = rs.normal(0.0, 0.0002)  # small drift
    sigma = 0.01
    prices = [base_price]
    for _ in range(1, length):
        prev = prices[-1]
        shock = rs.normal(mu * dt, sigma * math.sqrt(dt)) * prev
        prices.append(max(0.01, prev + shock))

    prices = np.array(prices, dtype=np.float32)
    # generate other technical-features as noisy transforms
    features = [prices]
    for i in range(n_features - 1):
        noise = rs.normal(0, 1.0, size=length).astype(np.float32)
        feat = (np.log1p(prices) * (0.5 + 0.1 * i)) + 0.01 * noise
        features.append(feat)

    seq = np.stack(features, axis=-1)
    return seq


def write_tfrecord_shards(output_dir: str,
                          prefix: str = 'massive',
                          n_shards: int = 10,
                          samples_per_shard: int = 1000,
                          seq_length: int = 128,
                          n_features: int = 16,
                          compress: bool = True,
                          seed: Optional[int] = None):
    """Generate sharded TFRecord files with synthetic sequences.

    Each example contains:
      - 'symbol': bytes (generated id)
      - 'sequence': float list (seq_length * n_features)
      - 'label': float (future return synthetic)

    The function streams data to disk and avoids large memory.
    """
    os.makedirs(output_dir, exist_ok=True)
    total = n_shards * samples_per_shard
    rng = np.random.RandomState(seed)

    for shard in range(n_shards):
        shard_path = os.path.join(output_dir, f"{prefix}-{shard:05d}.tfrecord")
        if compress:
            options = tf.io.TFRecordOptions(compression_type='GZIP')
            writer = tf.io.TFRecordWriter(shard_path + '.gz', options=options)
            target_path = shard_path + '.gz'
        else:
            writer = tf.io.TFRecordWriter(shard_path)
            target_path = shard_path

        for i in range(samples_per_shard):
            idx = shard * samples_per_shard + i
            symbol = f"SYM{idx % 1000:04d}"
            base_price = 50.0 + (idx % 200) * 0.1
            seq = synthetic_bar_sequence(seq_length, n_features, base_price=base_price, seed=int(rng.randint(0, 2**31)))
            # label: synthetic future log-return
            future = (seq[-1, 0] - seq[-2, 0]) / max(1e-6, seq[-2, 0])
            example = tf.train.Example(features=tf.train.Features(feature={
                'symbol': tf.train.Feature(bytes_list=tf.train.BytesList(value=[symbol.encode('utf-8')])),
                'sequence': _float_list_feature(seq.flatten().tolist()),
                'seq_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_length])),
                'n_features': tf.train.Feature(int64_list=tf.train.Int64List(value=[n_features])),
                'label': tf.train.Feature(float_list=tf.train.FloatList(value=[float(future)]))
            }))
            writer.write(example.SerializeToString())

        writer.close()
        print(f"Wrote shard {shard+1}/{n_shards}: {target_path}")


def estimate_dataset_size_bytes(n_shards: int, samples_per_shard: int, seq_length: int, n_features: int, compression_ratio: float = 5.0):
    """Rough estimate of bytes for planning. compression_ratio is approximate."""
    per_sample = seq_length * n_features * 4  # float32
    approx = n_shards * samples_per_shard * per_sample
    return int(approx / compression_ratio)


if __name__ == '__main__':
    # quick demo - small by default
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/massive', help='output directory')
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--samples-per-shard', type=int, default=32)
    parser.add_argument('--seq-length', type=int, default=64)
    parser.add_argument('--features', type=int, default=8)
    parser.add_argument('--no-compress', action='store_true')
    args = parser.parse_args()

    write_tfrecord_shards(args.out, n_shards=args.shards, samples_per_shard=args.samples_per_shard,
                          seq_length=args.seq_length, n_features=args.features, compress=not args.no_compress)
