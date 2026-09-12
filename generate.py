import sys

import numpy as np
import tensorflow as tf
from tokenizers import Tokenizer

from train_gpt2 import BLOCK_SIZE, WEIGHTS, build_model

# ---------------------------------------------------------------- settings --
PROMPT = "The history of"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8       # lower = safer, higher = more random
TOP_K = 40              # only sample from the 40 most likely tokens
END_OF_TEXT = 50256


def generate(model, tokenizer, prompt):
    ids = tokenizer.encode(prompt).ids

    for _ in range(MAX_NEW_TOKENS):
        context = np.array([ids[-BLOCK_SIZE:]], dtype=np.int32)
        logits = np.asarray(model(context))[0, -1].astype(np.float64) / TEMPERATURE

        # keep only the top-k tokens
        keep = np.argpartition(logits, -TOP_K)[-TOP_K:]
        filtered = np.full_like(logits, -np.inf)
        filtered[keep] = logits[keep]

        probs = np.exp(filtered - filtered.max())
        probs /= probs.sum()

        next_id = int(np.random.choice(len(probs), p=probs))
        if next_id == END_OF_TEXT:
            break
        ids.append(next_id)

    return tokenizer.decode(ids)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    if not WEIGHTS.exists():
        sys.exit("No trained weights. Run:  python train_gpt2.py")

    model = build_model()
    model.load_weights(str(WEIGHTS))

    print("\n" + "=" * 70)
    print(generate(model, Tokenizer.from_pretrained("gpt2"), PROMPT))
    print("=" * 70)
