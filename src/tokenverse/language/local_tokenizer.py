"""A tiny byte-level BPE tokenizer trained in memory, for offline use.

The tests, the figure script and the notebooks' offline fallback use this, so
nothing has to be downloaded. It's built with the same ``tokenizers`` machinery
as GPT-2's fast tokenizer, but trained on a handful of sentences, so its
vocabulary is tiny.

Requires the ``nlp`` extra.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

TINY_CORPUS: tuple[str, ...] = (
    "because the model reads tokens, not words",
    "because because because the tokenizer merges frequent pairs",
    "the cat sat because the mat was warm",
    "tokens become ids and ids become vectors",
    "hello world, hello tokens, hello vectors",
    "numbers like 2026 and 12580 are split into pieces",
    "the transformer reads the whole sequence",
    "lower lowest newer newest wider widest",
)


@lru_cache(maxsize=4)
def build_local_tokenizer(vocab_size: int = 320) -> Any:
    """Train and return a small ``PreTrainedTokenizerFast`` (byte-level BPE).

    The result is deterministic for a given ``vocab_size`` and cached.
    """
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
        from tokenizers.trainers import BpeTrainer
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            'The local tokenizer needs the NLP extra: pip install -e ".[nlp]"'
        ) from exc

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>"],
        show_progress=False,
    )
    tok.train_from_iterator(list(TINY_CORPUS), trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|endoftext|>",
        bos_token="<|endoftext|>",
    )
    fast.name_or_path = "local-tiny-bpe"
    return fast
