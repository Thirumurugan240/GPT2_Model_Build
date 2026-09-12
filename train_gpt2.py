import importlib.util
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


BLOCK_SIZE = 128       
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 3e-4

EMBED_DIM = 256        
NUM_HEADS = 4           
DFF = 1024             
NUM_LAYERS = 4         
DROPOUT = 0.1

VOCAB_SIZE = 50257      

BASE_DIR = Path(__file__).resolve().parent
TOKEN_DIR = BASE_DIR / "data" / "tokens"
WEIGHTS = BASE_DIR / "checkpoints" / "gpt2.weights.h5"



def load_tokens():
    """Read the .npy file that download_data.py created."""
    files = sorted(TOKEN_DIR.glob("*.npy"))
    if not files:
        sys.exit("No dataset found. Run:  python download_data.py")
    tokens = np.load(files[0])
    print(f"dataset : {files[0].name}  ({len(tokens):,} tokens)")
    return tokens


def make_dataset(tokens):
    """Cut the tokens into (input, target) pairs, where target is input shifted by one."""
    usable = (len(tokens) - 1) // BLOCK_SIZE * BLOCK_SIZE
    inputs = tokens[:usable].astype(np.int32).reshape(-1, BLOCK_SIZE)
    targets = tokens[1:usable + 1].astype(np.int32).reshape(-1, BLOCK_SIZE)

    split = int(len(inputs) * 0.95)
    train = tf.data.Dataset.from_tensor_slices((inputs[:split], targets[:split]))
    val = tf.data.Dataset.from_tensor_slices((inputs[split:], targets[split:]))

    train = train.shuffle(10_000).batch(BATCH_SIZE, drop_remainder=True)
    val = val.batch(BATCH_SIZE, drop_remainder=True)
    print(f"batches : {len(train)} train / {len(val)} validation")
    return train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE)


def build_model():
    """Import the GPT2 class from GPT-2.py (the hyphen blocks a normal import)."""
    spec = importlib.util.spec_from_file_location("gpt2_model", BASE_DIR / "GPT-2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.GPT2(
        vocab_size=VOCAB_SIZE,
        max_length=BLOCK_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dff=DFF,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT,
    )
    model(tf.zeros((1, BLOCK_SIZE), dtype=tf.int32))   

    params = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"model   : {params / 1e6:.1f}M parameters")
    return model


def main():
    train_ds, val_ds = make_dataset(load_tokens())
    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    WEIGHTS.parent.mkdir(exist_ok=True)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(str(WEIGHTS),
                                                      save_weights_only=True)],
    )

    model.save_weights(str(WEIGHTS))
    print(f"\nsaved weights to {WEIGHTS}")
    print("write text with:  python generate.py")


if __name__ == "__main__":
    main()
