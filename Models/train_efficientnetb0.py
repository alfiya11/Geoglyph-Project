#!/usr/bin/env python
"""
EfficientNet-B0 geoglyph classifier (two-stage training).

Converted from EfficientNetB0.ipynb to run as a Nautilus batch job.
The model, augmentation, loss, metrics, callbacks and two-stage strategy are
unchanged from the notebook. What changed:
  * Google Drive / Colab paths -> --data_dir / --output_dir (PVC mounts)
  * everything an experiment produces is written to ONE folder per run:
        <output_dir>/<run_name>/
  * the train/val/test split is created once and shared by every model
        <output_dir>/shared/data_splits.csv
  * hyperparameters are command-line arguments (one job per experiment)
  * ds.ignore_errors() now runs BEFORE batching (see note in make_dataset)

Usage (inside the batch job template):
    python train_efficientnetb0.py --data_dir /workspace/data \
        --output_dir /workspace/outputs --run_name effb0_run1
Quick pipeline test (tiny subset, 1 epoch per stage):
    python train_efficientnetb0.py --smoke_test --run_name test1
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

VALID_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def log(*a):
    print(*a, flush=True)  # flush so `kubectl logs` shows lines immediately


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/workspace/data",
                   help="folder that contains binary_dataset/{positive,negative}")
    p.add_argument("--output_dir", default="/workspace/outputs")
    p.add_argument("--run_name", default="efficientnetb0_baseline")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs_stage1", type=int, default=30)
    p.add_argument("--epochs_stage2", type=int, default=30)
    p.add_argument("--lr_stage1", type=float, default=1e-3)
    p.add_argument("--lr_stage2", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--dense_units", type=int, default=256)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke_test", action="store_true",
                   help="tiny subset, 1 epoch per stage, to test the pipeline")
    p.add_argument("--check_data", action="store_true",
                   help="count how many images each split really loads")
    p.add_argument("--allow_cpu", action="store_true",
                   help="do not fail if no GPU is visible")
    p.add_argument("--overwrite", action="store_true",
                   help="allow reusing a run_name that already has results")
    return p.parse_args()


# ---------------------------------------------------------------- data ------
def find_images(data_dir):
    """Scan binary_dataset/{positive,negative}. label 1 = geoglyph, 0 = background."""
    rows = []
    for sub, label in (("positive", 1), ("negative", 0)):
        folder = Path(data_dir) / "binary_dataset" / sub
        if not folder.is_dir():
            sys.exit(f"ERROR: folder not found: {folder}")
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in VALID_EXT:
                rows.append((f"binary_dataset/{sub}/{p.name}", label))
    return pd.DataFrame(rows, columns=["relpath", "label"])


def load_or_make_splits(df, shared_dir, seed):
    """Split ONCE (70/15/15, stratified) and reuse for every model so all
    experiments are directly comparable. Paths are stored relative to data_dir."""
    csv_path = Path(shared_dir) / "data_splits.csv"
    if csv_path.exists():
        split_df = pd.read_csv(csv_path)
        if set(split_df.relpath) != set(df.relpath):
            sys.exit(f"ERROR: {csv_path} does not match the images on disk. "
                     "If the dataset really changed, move that file aside on purpose "
                     "(it defines the split for ALL experiments).")
        log(f"Using existing split: {csv_path}")
        return split_df

    train_val_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=seed)
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.176, stratify=train_val_df["label"], random_state=seed)
    split_df = pd.concat([train_df.assign(split="train"),
                          val_df.assign(split="val"),
                          test_df.assign(split="test")]).reset_index(drop=True)
    Path(shared_dir).mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(f".tmp{os.getpid()}")
    split_df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)  # atomic: safe if two jobs start together
    log(f"Created new split: {csv_path}")
    return split_df


def make_dataset(dataframe, data_dir, img_size, batch_size, seed,
                 shuffle=False, with_path=False):
    """tf.data pipeline. EfficientNet normalizes internally, so pixels stay [0,255].

    NOTE: ignore_errors() comes BEFORE batch(). In the notebook it came after,
    which drops a whole batch of 32 if one image fails to load; before batch it
    skips only the bad image.
    """
    paths = [str(Path(data_dir) / p) for p in dataframe["relpath"]]
    labels = dataframe["label"].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), seed=seed,
                        reshuffle_each_iteration=True)

    def _load(path, label):
        raw = tf.io.read_file(path)
        image = tf.image.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32)
        return image, tf.cast(label, tf.float32), path

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.ignore_errors()
    if not with_path:
        ds = ds.map(lambda i, l, p: (i, l))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def count_samples(ds):
    return sum(int(b[0].shape[0]) for b in ds)


# --------------------------------------------------------------- model ------
def build_efficientnet_b0(img_size, dropout, dense_units, trainable_backbone=False):
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")
    x = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(factor=0.25, fill_mode="reflect"),
        layers.RandomBrightness(factor=0.15),
        layers.RandomContrast(factor=0.15),
        layers.RandomZoom(height_factor=(-0.1, 0.1), fill_mode="reflect"),
    ], name="augmentation")(inputs)
    backbone = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    backbone.trainable = trainable_backbone
    x = layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inputs, outputs, name="EfficientNetB0_geoglyph")


def compile_model(model, lr, label_smoothing):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc_roc", curve="ROC"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ])


def make_callbacks(ckpt_path, es_patience, min_lr):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path), monitor="val_auc_roc", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_auc_roc", mode="max", patience=es_patience,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_roc", mode="max", factor=0.5, patience=3,
            min_lr=min_lr, verbose=1),
    ]


# ------------------------------------------------------------- outputs ------
def plot_history(h1, h2, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, key in zip(axes, ["loss", "auc_roc", "recall"]):
        tr = list(h1.history[key]) + list(h2.history[key])
        va = list(h1.history["val_" + key]) + list(h2.history["val_" + key])
        ax.plot(tr, label="train")
        ax.plot(va, label="val")
        ax.axvline(len(h1.history[key]) - 0.5, color="gray", ls="--", lw=1)
        ax.set_title(key)
        ax.set_xlabel("epoch (dashed line = stage 1 -> 2)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["background", "geoglyph"])
    ax.set_yticklabels(["background", "geoglyph"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main ------
def main():
    args = parse_args()
    if args.smoke_test:
        args.epochs_stage1 = 1
        args.epochs_stage2 = 1
        if not args.run_name.startswith("smoke_"):
            args.run_name = "smoke_" + args.run_name

    run_dir = Path(args.output_dir) / args.run_name
    if (run_dir / "metrics.json").exists() and not args.overwrite:
        sys.exit(f"ERROR: {run_dir} already has results. Use a new --run_name "
                 "(or --overwrite) so earlier experiments are never replaced.")
    run_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = Path(args.output_dir) / "shared"

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    log("TensorFlow:", tf.__version__)
    log("GPU      :", gpus)
    if not gpus and not args.allow_cpu:
        sys.exit("ERROR: no GPU visible to TensorFlow. Check the image/CUDA setup "
                 "(or pass --allow_cpu for a slow CPU run).")
    log("Run      :", args.run_name, "->", run_dir)
    log("Git SHA  :", os.environ.get("GIT_SHA", "unknown"))
    t0 = time.time()

    # ---- data ----
    df = find_images(args.data_dir)
    log(f"Images found: {len(df)}  (positive {(df.label == 1).sum()}, "
        f"negative {(df.label == 0).sum()})")
    split_df = load_or_make_splits(df, shared_dir, args.seed)
    train_df = split_df[split_df.split == "train"].reset_index(drop=True)
    val_df = split_df[split_df.split == "val"].reset_index(drop=True)
    test_df = split_df[split_df.split == "test"].reset_index(drop=True)

    if args.smoke_test:
        def sub(d, n):
            return pd.concat([g.sample(min(len(g), n), random_state=args.seed)
                              for _, g in d.groupby("label")]).reset_index(drop=True)
        train_df, val_df, test_df = sub(train_df, 16), sub(val_df, 8), sub(test_df, 8)
        log("SMOKE TEST: tiny subset, 1 epoch per stage")

    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        log(f"  {name:<5}: {len(d):>5} images  (positive {(d.label == 1).sum()})")

    cw = compute_class_weight("balanced", classes=np.array([0, 1]),
                              y=train_df["label"].values)
    class_weight_dict = {0: cw[0], 1: cw[1]}
    log(f"Class weights: {class_weight_dict}")

    mk = dict(data_dir=args.data_dir, img_size=args.img_size,
              batch_size=args.batch_size, seed=args.seed)
    train_ds = make_dataset(train_df, shuffle=True, **mk)
    val_ds = make_dataset(val_df, **mk)
    test_ds = make_dataset(test_df, **mk)
    test_ds_paths = make_dataset(test_df, with_path=True, **mk)

    if args.check_data or args.smoke_test:
        for name, ds, d in (("train", train_ds, train_df), ("val", val_ds, val_df),
                            ("test", test_ds, test_df)):
            n = count_samples(ds)
            flag = "" if n == len(d) else "   <-- SOME IMAGES FAILED TO LOAD"
            log(f"  loaded {name:<5}: {n} / {len(d)}{flag}")
    n_test_loaded = count_samples(test_ds)
    if n_test_loaded != len(test_df):
        log(f"WARNING: only {n_test_loaded} of {len(test_df)} test images loaded.")

    # ---- save the config (reproducibility record) ----
    config = vars(args) | {
        "git_sha": os.environ.get("GIT_SHA", "unknown"),
        "tensorflow": tf.__version__,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "class_weights": {str(k): float(v) for k, v in class_weight_dict.items()},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ---- stage 1: frozen backbone ----
    model = build_efficientnet_b0(args.img_size, args.dropout, args.dense_units,
                                  trainable_backbone=False)
    compile_model(model, args.lr_stage1, args.label_smoothing)
    ckpt1 = run_dir / "stage1_best.weights.h5"
    log("\n=== Stage 1: frozen backbone ===")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_stage1,
                   class_weight=class_weight_dict,
                   callbacks=make_callbacks(ckpt1, es_patience=7, min_lr=1e-6),
                   verbose=2)
    pd.DataFrame(h1.history).to_csv(run_dir / "history_stage1.csv", index=False)
    if ckpt1.exists():
        model.load_weights(ckpt1)

    # ---- stage 2: unfreeze everything except BatchNorm ----
    for layer in model.layers:
        layer.trainable = not isinstance(layer, keras.layers.BatchNormalization)
    compile_model(model, args.lr_stage2, args.label_smoothing)
    ckpt2 = run_dir / "stage2_best.weights.h5"
    log("\n=== Stage 2: fine-tuning (BatchNorm frozen) ===")
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_stage2,
                   class_weight=class_weight_dict,
                   callbacks=make_callbacks(ckpt2, es_patience=8, min_lr=1e-7),
                   verbose=2)
    pd.DataFrame(h2.history).to_csv(run_dir / "history_stage2.csv", index=False)
    if ckpt2.exists():
        model.load_weights(ckpt2)
    plot_history(h1, h2, run_dir / "training_curves.png")

    # ---- final evaluation on the held-out test set (look at this ONCE) ----
    log("\n=== Test-set evaluation ===")
    keras_res = model.evaluate(test_ds, return_dict=True, verbose=2)

    y_true, y_prob, paths = [], [], []
    for x, y, p in test_ds_paths:
        y_prob.extend(model(x, training=False).numpy().ravel().tolist())
        y_true.extend(y.numpy().astype(int).tolist())
        paths.extend(pp.decode() for pp in p.numpy())
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame({"filepath": paths, "label": y_true, "prob_geoglyph": y_prob,
                  "pred_at_0.5": y_pred}).to_csv(run_dir / "test_predictions.csv",
                                                 index=False)
    plot_confusion(cm, run_dir / "confusion_matrix.png")

    metrics = {
        "run_name": args.run_name,
        "n_test_evaluated": int(len(y_true)),
        "keras": {k: float(v) for k, v in keras_res.items()},
        "sklearn_threshold_0.5": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "auc_roc": float(roc_auc_score(y_true, y_prob)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        },
        "confusion_matrix [[TN, FP], [FN, TP]]": cm.tolist(),
        "elapsed_minutes": round((time.time() - t0) / 60, 1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    log("=" * 50)
    log(f"  {args.run_name} - FINAL TEST RESULTS")
    log("=" * 50)
    for k, v in metrics["sklearn_threshold_0.5"].items():
        log(f"  {k:<10}: {v:.4f}")
    log(f"  confusion [[TN, FP], [FN, TP]]: {cm.tolist()}")
    log(f"  results saved in: {run_dir}")
    log("=" * 50)


if __name__ == "__main__":
    main()
