"""Load the pretrained Hugging Face tokenizers used in the notebooks.

Only the tokenizer files are downloaded, not the model weights. Each tokenizer
has its own vocabulary, so token ID 1000 in GPT-2 has nothing to do with token
ID 1000 in BERT or T5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenizerInfo:
    """Short description of a supported pretrained tokenizer."""

    model_name: str
    family: str
    notes: str


SUPPORTED_TOKENIZERS: dict[str, TokenizerInfo] = {
    "gpt2": TokenizerInfo(
        "gpt2",
        "Byte-level BPE",
        "Visible token strings can look odd (e.g. 'Ġ' for a leading space) "
        "because BPE runs over a byte-to-Unicode mapping.",
    ),
    "bert-base-uncased": TokenizerInfo(
        "bert-base-uncased",
        "WordPiece",
        "Lowercases and strips accents, then splits words into pieces; "
        "continuation pieces start with '##'.",
    ),
    "t5-small": TokenizerInfo(
        "t5-small",
        "SentencePiece Unigram",
        "Chooses a segmentation from a probabilistic unigram vocabulary; "
        "'▁' marks a preceding space.",
    ),
}


class TokenizerLoadError(RuntimeError):
    """Raised when a pretrained tokenizer cannot be loaded."""


_CACHE: dict[tuple[str, bool], Any] = {}


def clear_tokenizer_cache() -> None:
    """Forget tokenizers loaded earlier in this process."""
    _CACHE.clear()


def load_tokenizer(model_name: str, *, local_files_only: bool = False) -> Any:
    """Load a fast Hugging Face tokenizer by model identifier.

    Only tokenizer files (vocabulary, merges, configuration) are downloaded.
    Results are cached per ``(model_name, local_files_only)``.

    Args:
        model_name: A Hub identifier such as ``"gpt2"``, ``"bert-base-uncased"``
            or ``"t5-small"``. Other identifiers are allowed but unsupported.
        local_files_only: If ``True``, only use files already in the local
            Hugging Face cache (useful offline).

    Returns:
        A ``PreTrainedTokenizerFast`` instance.

    Raises:
        TokenizerLoadError: If Transformers or SentencePiece is missing, the
            files cannot be downloaded or found, or the loaded tokenizer is not
            a fast tokenizer.
    """
    key = (model_name, local_files_only)
    if key in _CACHE:
        return _CACHE[key]

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise TokenizerLoadError(
            "Hugging Face Transformers is not installed. Install the NLP extra "
            'with: pip install -e ".[nlp]"'
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
    except ImportError as exc:
        raise TokenizerLoadError(
            f"Loading '{model_name}' needs a library that is not installed "
            f"({exc}). T5 needs SentencePiece: pip install sentencepiece"
        ) from exc
    except (OSError, ValueError) as exc:
        where = (
            "in the local Hugging Face cache (local_files_only=True)"
            if local_files_only
            else "on the Hugging Face Hub (check your internet connection)"
        )
        raise TokenizerLoadError(
            f"Could not load tokenizer files for '{model_name}' {where}. "
            f"Original error: {exc}"
        ) from exc

    if not getattr(tokenizer, "is_fast", False):
        raise TokenizerLoadError(
            f"'{model_name}' loaded a slow tokenizer; offset mappings need a fast "
            "(Rust-backed) tokenizer. Make sure the 'tokenizers' package is "
            "installed."
        )

    _CACHE[key] = tokenizer
    return tokenizer


def load_supported_tokenizers(
    *, local_files_only: bool = False
) -> tuple[dict[str, Any], dict[str, str]]:
    """Try to load every tokenizer in :data:`SUPPORTED_TOKENIZERS`.

    Returns:
        ``(loaded, errors)``: tokenizers that loaded, keyed by model name, and
        error messages for those that did not. Useful in notebooks so one
        failed download does not stop the rest.
    """
    loaded: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name in SUPPORTED_TOKENIZERS:
        try:
            loaded[name] = load_tokenizer(name, local_files_only=local_files_only)
        except TokenizerLoadError as exc:
            errors[name] = str(exc)
    return loaded, errors
