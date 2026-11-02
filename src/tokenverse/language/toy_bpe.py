"""A small Byte-Pair Encoding (BPE) trainer you can watch merge by merge.

Character-level BPE in the style of Sennrich et al. (2016). GPT-2's tokenizer
adds a byte-level alphabet, regex pre-tokenisation and special tokens on top of
this; none of that is here.

How it works:

1. Split the corpus on whitespace into words and count each word.
2. Write each word as its characters followed by an end-of-word marker
   ``</w>``, e.g. ``low`` -> ``l o w </w>``.
3. Count adjacent symbol pairs across all words, weighted by word frequency.
4. Merge the most frequent pair everywhere it occurs (left to right,
   non-overlapping). Ties go to the lexicographically smallest pair so results
   are deterministic.
5. Repeat for ``num_merges`` steps or until no pairs remain.

Encoding new text applies the learned merges in the order they were learned.
Characters never seen during training stay as single symbols.

Whitespace: text is split with :meth:`str.split`, so runs of spaces, tabs and
newlines all collapse to a single space after :meth:`ToyBPE.decode`. Only the
sequence of whitespace-separated words round-trips, not the exact whitespace.

Example:
    >>> bpe = ToyBPE()
    >>> bpe.train("low lower lowest low", num_merges=6)
    >>> tokens = bpe.encode("flower")
    >>> bpe.decode(tokens)
    'lower newer'
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

END_OF_WORD = "</w>"

Pair = tuple[str, str]
Word = tuple[str, ...]


@dataclass(frozen=True)
class MergeStep:
    """One step of BPE training.

    Attributes:
        step: 1-based merge number.
        pair: The two symbols that were merged.
        merged: The new symbol (``pair[0] + pair[1]``).
        frequency: How often the pair occurred, weighted by word frequency,
            just before the merge.
        samples: Segmentations of a few of the most frequent training words
            right after this merge, as ``{word: [symbols]}``.
    """

    step: int
    pair: Pair
    merged: str
    frequency: int
    samples: dict[str, list[str]]


def _word_to_symbols(word: str) -> Word:
    return (*word, END_OF_WORD)


def _count_pairs(word_counts: Counter[Word]) -> Counter[Pair]:
    pairs: Counter[Pair] = Counter()
    for symbols, count in word_counts.items():
        for a, b in zip(symbols, symbols[1:], strict=False):
            pairs[(a, b)] += count
    return pairs


def _merge_symbols(symbols: Word, pair: Pair) -> Word:
    """Merge every non-overlapping occurrence of ``pair``, scanning left to right."""
    merged: list[str] = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


@dataclass
class ToyBPE:
    """Character-level BPE.

    Attributes:
        merges: Learned merges in priority order.
        merge_history: A :class:`MergeStep` for each merge, for inspection.
        vocab: Symbols known after training (base characters, ``</w>`` and
            every merged symbol), in the order they were added.
        num_samples: How many frequent words to record in each history step.
    """

    num_samples: int = 4
    merges: list[Pair] = field(default_factory=list)
    merge_history: list[MergeStep] = field(default_factory=list)
    vocab: list[str] = field(default_factory=list)

    def train(self, corpus: str | Iterable[str], num_merges: int = 10) -> ToyBPE:
        """Learn up to ``num_merges`` merges from ``corpus``.

        Args:
            corpus: A string, or an iterable of strings (e.g. lines).
            num_merges: Maximum number of merges. Training stops early if no
                adjacent pairs remain.

        Returns:
            ``self``, so calls can be chained.
        """
        if num_merges < 0:
            raise ValueError("num_merges must be zero or positive")
        text = corpus if isinstance(corpus, str) else " ".join(corpus)
        word_freq = Counter(text.split())
        word_counts: Counter[Word] = Counter(
            {_word_to_symbols(w): c for w, c in word_freq.items()}
        )

        self.merges = []
        self.merge_history = []
        base = sorted({ch for w in word_freq for ch in w})
        self.vocab = [*base, END_OF_WORD] if word_freq else []

        sample_words = [w for w, _ in word_freq.most_common(self.num_samples)]

        for step in range(1, num_merges + 1):
            pairs = _count_pairs(word_counts)
            if not pairs:
                break
            # Highest count first; ties broken by the smallest pair.
            pair, freq = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))
            new_counts: Counter[Word] = Counter()
            for symbols, count in word_counts.items():
                new_counts[_merge_symbols(symbols, pair)] += count
            word_counts = new_counts

            merged = pair[0] + pair[1]
            self.merges.append(pair)
            if merged not in self.vocab:
                self.vocab.append(merged)
            self.merge_history.append(
                MergeStep(
                    step=step,
                    pair=pair,
                    merged=merged,
                    frequency=freq,
                    samples={w: self.encode_word(w) for w in sample_words},
                )
            )
        return self

    def encode_word(self, word: str) -> list[str]:
        """Segment one word (no whitespace) using the learned merges."""
        if not word:
            return []
        symbols = _word_to_symbols(word)
        ranks = {pair: i for i, pair in enumerate(self.merges)}
        while len(symbols) > 1:
            candidates = [
                (ranks[p], p)
                for p in zip(symbols, symbols[1:], strict=False)
                if p in ranks
            ]
            if not candidates:
                break
            _, best = min(candidates)
            symbols = _merge_symbols(symbols, best)
        return list(symbols)

    def encode(self, text: str) -> list[str]:
        """Split ``text`` on whitespace and segment each word.

        Unseen characters are kept as single-character symbols. Empty or
        whitespace-only input returns ``[]``.
        """
        tokens: list[str] = []
        for word in text.split():
            tokens.extend(self.encode_word(word))
        return tokens

    def decode(self, tokens: Iterable[str]) -> str:
        """Join symbols back into words separated by single spaces."""
        joined = "".join(tokens)
        words = [w for w in joined.split(END_OF_WORD) if w]
        return " ".join(words)

    def segment_corpus(self, corpus: str) -> dict[str, list[str]]:
        """Current segmentation of every distinct word in ``corpus``."""
        return {w: self.encode_word(w) for w in dict.fromkeys(corpus.split())}
