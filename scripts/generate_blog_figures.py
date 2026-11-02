"""Regenerate the seven article figures in artifacts/figures/.

Runs offline: it uses the toy BPE, the in-memory tiny tokenizer and a
synthetic image, so nothing is downloaded.

    python scripts/generate_blog_figures.py
    python scripts/generate_blog_figures.py --output-dir /tmp/figs --dpi 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tokenverse.language.boundary_analysis import inspect_boundary  # noqa: E402
from tokenverse.language.local_tokenizer import build_local_tokenizer  # noqa: E402
from tokenverse.language.number_tokenization import ARITHMETIC_EXAMPLE  # noqa: E402
from tokenverse.language.toy_bpe import ToyBPE  # noqa: E402
from tokenverse.visualization import text_visualization as tv  # noqa: E402
from tokenverse.visualization import vision_visualization as vv  # noqa: E402

# The corpus from the article's BPE walkthrough.
BPE_CORPUS = "low lower lowest low"

# GPT-2's real output for the first sentence of the quote. It's written out
# here so the script stays offline; Notebook 1 shows the same values live.
PIPELINE_TEXT = "Time is an illusion."
PIPELINE_TOKENS = ["Time", " is", " an", " illusion", "."]
PIPELINE_IDS = [7575, 318, 281, 17878, 13]


def build_figures() -> dict[str, plt.Figure]:
    """Create every figure and return them keyed by filename."""
    figures: dict[str, plt.Figure] = {}

    tokenizer = build_local_tokenizer()
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(len(PIPELINE_IDS), 6))
    figures["tokenization_pipeline.png"] = tv.plot_tokenization_pipeline(
        PIPELINE_TOKENS, PIPELINE_IDS, embeddings, PIPELINE_TEXT
    )

    bpe = ToyBPE().train(BPE_CORPUS, num_merges=6)
    figures["bpe_merges.png"] = tv.plot_bpe_merges(bpe, ["low", "lower", "lowest"])

    figures["number_tokenization.png"] = tv.plot_number_groupings(ARITHMETIC_EXAMPLE)

    # Real offsets from the tiny local tokenizer.
    boundary = inspect_boundary("becau", "se", tokenizer)
    figures["partial_token_boundary.png"] = tv.plot_boundary(boundary)

    from tokenverse.vision.image_utils import make_sample_image

    image = make_sample_image(224)
    fig = vv.plot_patch_grid(image, 16)
    fig.tight_layout()
    figures["image_patchification.png"] = fig
    figures["vit_cls_sequence.png"] = vv.plot_cls_sequence(196)
    figures["vit_architecture.png"] = vv.plot_vit_architecture()
    return figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the article figures")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    parser.add_argument(
        "--dpi", type=int, default=140, help="Resolution (default 140, ~1400px wide)"
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in build_figures().items():
        path = args.output_dir / name
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
