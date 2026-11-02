# The Hitchhiker's Guide to Tokens

> "Time is an illusion. Lunchtime doubly so."
>
> Douglas Adams, *The Hitchhiker's Guide to the Galaxy*

This is the code that goes with my article *The Hitchhiker's Guide to
Tokens*, a journey through tokenisation, numbers, image patches and the
little `[CLS]` token. The article explains the ideas. This repository lets you
poke at them.

The easiest way in is through the three notebooks. Each one runs in Google
Colab from a fresh runtime, so there's nothing to install.

| | Notebook | What's in it |
|---|---|---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srishtiii28/hitchhiker_tokens/blob/main/notebooks/01_understanding_tokenization.ipynb) | **1. What exactly is a token?** | characters, words and subwords; token IDs vs embeddings; BPE by hand; GPT-2 vs BERT vs T5 on the Adams quote |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srishtiii28/hitchhiker_tokens/blob/main/notebooks/02_numbers_and_token_boundaries.ipynb) | **2. The number problem, and when token boundaries get in the way** | how GPT-2 splits numbers; left-to-right vs right-to-left digit grouping; the Partial Token and Prompt Boundary Problems; UTF-8 bytes |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srishtiii28/hitchhiker_tokens/blob/main/notebooks/03_vision_transformers_and_cls.ipynb) | **3. A picture is worth 16 × 16 tokens** | patches, patch embeddings, positional embeddings, the `[CLS]` token, a mini ViT and a tiny training run |

The first cell of each notebook clones this repository and installs it. No API
keys, Google Drive or GPU needed. Notebooks 1 and 2 download the GPT-2, BERT and
T5 tokenizer files (a few MB, no model weights). If that fails, they carry on
with a small tokenizer trained on the spot.

## Running it locally

You'll need Python 3.11 or newer.

```bash
git clone https://github.com/srishtiii28/hitchhiker_tokens.git
cd hitchhiker_tokens
python -m venv .venv
source .venv/bin/activate
pip install -e ".[nlp,notebooks,vision,dev]"
jupyter lab notebooks/
```

The extras are optional. The base install is enough for the toy BPE, the digit
grouping and the UTF-8 tools; `nlp` adds the Hugging Face tokenizers, `vision`
adds PyTorch, `notebooks` adds Jupyter and `dev` adds pytest and ruff.

## Scripts

```bash
# See how a tokenizer splits some text
python scripts/inspect_tokenizer.py --text "Time is an illusion. Lunchtime doubly so."
python scripts/inspect_tokenizer.py --model t5-small --text "Broken bottles break."
python scripts/inspect_tokenizer.py --model bert-base-uncased --input-file notes.txt

# Print every tensor shape in a forward pass of the mini ViT
python scripts/run_vit_demo.py

# Train the mini ViT on circles, squares and triangles (about 10 s on a laptop)
python scripts/train_toy_vit.py

# Redraw the figures in artifacts/figures/
python scripts/generate_blog_figures.py
```

`train_toy_vit.py` writes `metrics.csv`, `training_curve.png` and `model.pt` to
`artifacts/training_runs/toy_vit/`. Pass `--no-checkpoint` to skip the model
file.

## What's where

```text
src/tokenverse/
├── language/
│   ├── tokenizer_loader.py      # load the gpt2, bert-base-uncased and t5-small tokenizers
│   ├── tokenizer_inspector.py   # inspect_text, compare_tokenizers
│   ├── toy_bpe.py               # a small BPE you can watch merge by merge
│   ├── number_tokenization.py   # single-digit, L2R and R2L grouping
│   ├── boundary_analysis.py     # where a prompt boundary cuts through a token
│   ├── byte_analysis.py         # code points vs UTF-8 bytes
│   └── local_tokenizer.py       # tiny BPE trained in memory, for offline use
├── vision/
│   ├── image_utils.py
│   ├── patch_embedding.py       # patchify, PatchEmbeddingWithCLS
│   ├── mini_vit.py              # ViTConfig, MiniViT
│   └── synthetic_dataset.py     # circles, squares and triangles
└── visualization/               # the plots used in the notebooks and article
notebooks/                       # the three notebooks
scripts/                         # command-line versions of the above
tests/                           # pytest suite, runs offline
artifacts/figures/               # the article's figures
```

## The mini ViT, shape by shape

With the defaults from the article (224 × 224 images, 16 × 16 patches,
dimension 192, depth 4, 3 heads, 10 classes) and a batch of two:

| Stage | Shape |
|---|---|
| Input images | `[2, 3, 224, 224]` |
| Patch projection | `[2, 192, 14, 14]` |
| Patch embeddings | `[2, 196, 192]` |
| CLS + patch sequence | `[2, 197, 192]` |
| Transformer encoder | `[2, 197, 192]` |
| CLS representation | `[2, 192]` |
| Classification logits | `[2, 10]` |

## Tests

```bash
pytest -q
ruff check .
ruff format --check .
```

The tests never touch the network. They use a tokenizer trained in memory and
mock the Hugging Face downloads.

## Notes

- `ToyBPE` works on characters, not bytes, so it isn't GPT-2's tokenizer. It's
  there to make the merges visible.
- The boundary check looks at token offsets only. It doesn't compute model
  probabilities, and ByteSampler is explained in the notebooks but not
  implemented.
- The mini ViT is untrained unless you run the toy training, and the toy task
  says nothing about real image classification.

## References

1. [From Digits to Decisions: How Tokenization Impacts Arithmetic in LLMs](https://huggingface.co/spaces/huggingface/number-tokenization-blog), Hugging Face
2. [How does AI think? In token-language](https://substack.com/@practicalwisteria/p-196978882), Pamela Wang
3. [How do language models memorize?](https://softmaxdata.com/blog/technical-deep-dive-how-do-language-models-memorize/), Softmax Data
4. [Compute Optimal Tokenization](https://arxiv.org/abs/2605.01188), arXiv:2605.01188
5. [Sampling from Your Language Model One Byte at a Time](https://arxiv.org/abs/2506.14123), arXiv:2506.14123
6. [Are You Going to Finish That? A Practical Study of the Partial Token Problem](https://arxiv.org/abs/2601.23223), arXiv:2601.23223
7. [Broken Tokens? Your Language Model Can Secretly Handle Non-Canonical Tokenizations](https://arxiv.org/abs/2506.19004), arXiv:2506.19004
8. [An Image Is Worth 16 × 16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), Dosovitskiy et al.

Released under the [MIT License](LICENSE).
