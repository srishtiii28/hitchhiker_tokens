"""Print how a pretrained tokenizer splits some text.

Examples:
    python scripts/inspect_tokenizer.py --text "Hello, world!"
    python scripts/inspect_tokenizer.py --model bert-base-uncased --text "café"
    python scripts/inspect_tokenizer.py --model t5-small --input-file notes.txt
    python scripts/inspect_tokenizer.py --text "Hi" --special-tokens

Downloads tokenizer files (not model weights) on first use. Add --offline to
use only files already in the Hugging Face cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tokenverse.language.tokenizer_inspector import inspect_text
from tokenverse.language.tokenizer_loader import (
    SUPPORTED_TOKENIZERS,
    TokenizerLoadError,
    load_tokenizer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gpt2", help="Hub id (default: gpt2)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Text to tokenise")
    src.add_argument("--input-file", type=Path, help="UTF-8 text file to tokenise")
    parser.add_argument(
        "--special-tokens",
        action="store_true",
        help="Add the tokenizer's special tokens (e.g. BERT's [CLS]/[SEP])",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Only use locally cached files"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input_file is not None:
        try:
            text = args.input_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"error: file not found: {args.input_file}", file=sys.stderr)
            return 2
        except UnicodeDecodeError as exc:
            print(
                f"error: {args.input_file} is not valid UTF-8: {exc}", file=sys.stderr
            )
            return 2
    else:
        text = args.text

    try:
        tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    except TokenizerLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = inspect_text(text, tokenizer, add_special_tokens=args.special_tokens)
    info = SUPPORTED_TOKENIZERS.get(args.model)
    family = f" ({info.family})" if info else ""
    print(f"Tokenizer : {args.model}{family}")
    print(f"Input     : {text!r}")
    print(
        f"Tokens    : {report.num_tokens}   code points: {report.num_code_points}   "
        f"code points/token: "
        + (f"{report.chars_per_token:.2f}" if report.chars_per_token else "n/a")
    )
    print()
    print(report.format_table())
    print()
    print(f"Decoded   : {report.decoded!r}")
    print(f"Round trip: {'exact' if report.round_trip else 'not exact'}")
    if not report.round_trip:
        print(
            "            (normalising tokenizers such as bert-base-uncased lowercase "
            "and strip accents, so the decoded text can differ)"
        )
    print("Note      : token IDs are specific to this tokenizer's vocabulary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
