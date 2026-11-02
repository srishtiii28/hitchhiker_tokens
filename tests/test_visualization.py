"""Smoke tests: every figure renders offline without errors."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from tokenverse.visualization._theme import plot_safe  # noqa: E402


def test_plot_safe_replaces_missing_glyphs():
    assert plot_safe("café") == "café"
    assert plot_safe("你") == "U+4F60"
    assert plot_safe("a b") == "a b"


def test_blog_figures_render(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import importlib.util
    from pathlib import Path

    script = Path(__file__).parents[1] / "scripts" / "generate_blog_figures.py"
    spec = importlib.util.spec_from_file_location("generate_blog_figures", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--output-dir", str(tmp_path), "--dpi", "40"]) == 0
    names = {p.name for p in tmp_path.glob("*.png")}
    assert names == {
        "tokenization_pipeline.png",
        "bpe_merges.png",
        "number_tokenization.png",
        "partial_token_boundary.png",
        "image_patchification.png",
        "vit_cls_sequence.png",
        "vit_architecture.png",
    }
    plt.close("all")


def test_text_plots_render(local_tokenizer):
    from tokenverse.language import compare_tokenizers, inspect_text
    from tokenverse.visualization import text_visualization as tv

    comparison = compare_tokenizers("hello 你好", {"tiny": local_tokenizer})
    tv.plot_token_overlay(inspect_text("hello world", local_tokenizer))
    tv.plot_tokenizer_rows(comparison)
    tv.plot_token_counts(comparison)
    tv.plot_number_groupings(["12345", "-1,234.5"])
    tv.plot_utf8_bytes("Aé你🙂")
    plt.close("all")
