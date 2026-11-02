"""Dark-themed Matplotlib figures, kept separate from tokenizer and model logic.

``text_visualization`` needs only the base dependencies. ``vision_visualization``
works on PIL images or tensors and does not import PyTorch itself.
"""

from tokenverse.visualization._theme import PALETTE, ROLE_COLOURS, apply_dark_theme

__all__ = ["PALETTE", "ROLE_COLOURS", "apply_dark_theme"]
