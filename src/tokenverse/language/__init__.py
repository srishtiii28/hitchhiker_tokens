"""Language-side tools: tokenizer inspection, toy BPE, numbers, boundaries, bytes.

The pure-Python modules (``toy_bpe``, ``number_tokenization``,
``boundary_analysis``, ``byte_analysis``) only need the standard library.
``tokenizer_loader`` imports Hugging Face Transformers lazily, so importing
this package does not require the ``nlp`` extra.
"""

from tokenverse.language.boundary_analysis import (
    BoundaryReport,
    find_crossing_tokens,
    inspect_boundary,
)
from tokenverse.language.byte_analysis import ByteRecord, inspect_bytes
from tokenverse.language.number_tokenization import (
    ARITHMETIC_EXAMPLE,
    segment_expression,
    segment_number,
)
from tokenverse.language.tokenizer_inspector import (
    EXAMPLES,
    TokenizationReport,
    compare_tokenizers,
    inspect_text,
)
from tokenverse.language.tokenizer_loader import (
    SUPPORTED_TOKENIZERS,
    TokenizerLoadError,
    load_tokenizer,
)
from tokenverse.language.toy_bpe import MergeStep, ToyBPE

__all__ = [
    "ARITHMETIC_EXAMPLE",
    "EXAMPLES",
    "SUPPORTED_TOKENIZERS",
    "BoundaryReport",
    "ByteRecord",
    "MergeStep",
    "TokenizationReport",
    "TokenizerLoadError",
    "ToyBPE",
    "compare_tokenizers",
    "find_crossing_tokens",
    "inspect_boundary",
    "inspect_bytes",
    "inspect_text",
    "load_tokenizer",
    "segment_expression",
    "segment_number",
]
