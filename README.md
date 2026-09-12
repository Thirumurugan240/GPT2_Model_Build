# GPT-2 from scratch, in TensorFlow

A small GPT-2 language model written from scratch with Keras layers, trained on
real text from Hugging Face. No pretrained weights are used anywhere. The model
learns to write from nothing.

---

## 1. Install

Python 3.11 is what this was built against.

```bash
pip install -r requirements.txt
```

On Windows, TensorFlow runs on CPU only. For GPU training use Linux, WSL2, or
Google Colab.

---

## 2. Get the dataset

```bash
python download_data.py
```

This downloads WikiText-2 from Hugging Face, converts it into GPT-2 tokens, and
saves them. It takes under a minute and only needs to be run once.

It writes two things:

| Folder | What is in it |
| --- | --- |
| `data/raw/` | the original parquet file from Hugging Face, about 6 MB |
| `data/tokens/` | 2,000,000 token ids saved as a `.npy` file, about 4 MB |

If the token file already exists the script prints a message and exits without
downloading anything.

---

## 3. Train

```bash
python train_gpt2.py
```

The trainer loads the tokens, cuts them into 15,624 sequences of 128 tokens,
keeps 95 percent for training and 5 percent for validation, then trains the
model to predict the next token at every position.

Weights are saved to `checkpoints/gpt2.weights.h5` after every epoch, so an
interrupted run does not lose everything.

Training on a CPU is slow. Start with the settings as they are and only make the
model bigger once you have seen one full run finish.

---

## 4. Write text

```bash
python generate.py
```

This loads the trained weights and writes 100 tokens starting from a prompt.
Open the file and change `PROMPT` to write about something else.

The output will look like nonsense at first. A model this small, trained on two
million tokens for three epochs, learns spelling and common word pairs long
before it learns to make sense.

---

## Files

| File | What it does |
| --- | --- |
| `GPT-2.py` | the model itself: attention, feed-forward, transformer blocks |
| `download_data.py` | downloads and tokenizes the dataset |
| `train_gpt2.py` | loads the tokens and trains the model |
| `generate.py` | writes text with the trained model |
| `model.ipynb` | Learning Version |
| `requirements.txt` | the packages you need |

---

## Settings you can change

Every setting is a constant at the top of its file. There are no command-line
flags to learn.

In `train_gpt2.py`:

| Setting | Default | What it controls |
| --- | --- | --- |
| `BLOCK_SIZE` | 128 | how many tokens the model sees at once |
| `BATCH_SIZE` | 8 | sequences trained on at the same time |
| `EPOCHS` | 3 | passes over the whole dataset |
| `LEARNING_RATE` | 0.0003 | how big a step each update takes |
| `EMBED_DIM` | 256 | width of the model |
| `NUM_HEADS` | 4 | attention heads, must divide `EMBED_DIM` |
| `DFF` | 1024 | width of the feed-forward layer |
| `NUM_LAYERS` | 4 | stacked transformer blocks |

In `generate.py`:

| Setting | Default | What it controls |
| --- | --- | --- |
| `PROMPT` | "The history of" | the text the model continues |
| `MAX_NEW_TOKENS` | 100 | how much to write |
| `TEMPERATURE` | 0.8 | lower is repetitive, higher is chaotic |
| `TOP_K` | 40 | only sample from the 40 most likely next tokens |

If you change `EMBED_DIM`, `NUM_LAYERS`, or any other model setting, delete
`checkpoints/` first. Old weights do not fit a differently shaped model.

---

## Using a different dataset

Open `download_data.py` and change the settings at the top, then delete the old
file in `data/tokens/` and run it again.

```python
DATASET = "roneneldan/TinyStories"
CONFIG = ""
```

Any Hugging Face dataset that stores parquet files and has a text column will
work. TinyStories is a good second choice, because simple children's stories are
much easier for a small model to learn than encyclopedia articles.

---

## Model sizes

The default settings build a model of about 29 million parameters, which is
small enough to train on a laptop.

| Configuration | Parameters |
| --- | --- |
| the defaults in `train_gpt2.py` | 29 million |


---

## When something goes wrong

**No dataset found.** Run `python download_data.py` first.

**No trained weights.** Run `python train_gpt2.py` before `python generate.py`.

**Out of memory.** Lower `BATCH_SIZE` to 4 or 2, then lower `BLOCK_SIZE` to 64.

**Shape mismatch when loading weights.** The model settings changed since the
weights were saved. Delete `checkpoints/` and train again.

**Training is far too slow.** Halve `NUM_LAYERS` and `EMBED_DIM`, or move to
Google Colab and pick a GPU runtime.
