"""The Hitchhiker's Guide to the Tokenverse.

Companion code for the article of the same name: tokenisation, numbers,
token boundaries, image patches and the Vision Transformer ``[CLS]`` token.

Subpackages:

- :mod:`tokenverse.language` -- tokenizer inspection, a toy BPE, number
  segmentation, boundary diagnostics and UTF-8 inspection. Only the pretrained
  tokenizer helpers need the ``nlp`` extra.
- :mod:`tokenverse.vision` -- patchification, patch embeddings with a learnable
  ``[CLS]`` token, a miniature ViT and a synthetic shapes dataset. Needs the
  ``vision`` extra (PyTorch).
- :mod:`tokenverse.visualization` -- dark-themed Matplotlib figures.
"""

__version__ = "0.1.0"
