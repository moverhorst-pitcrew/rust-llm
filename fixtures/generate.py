#!/usr/bin/env python3
"""Regenerates the tokenizer and chat-template fixtures.

The point of these fixtures is that CI checks exact token ids and exact prompts
without the weights, without network, and without trusting our own Rust code:
the ids come from the Hugging Face `tokenizers` package and the prompts from
the model's own `chat_template.jinja` rendered by Jinja2.

    python3 -m venv .venv && .venv/bin/pip install tokenizers jinja2
    .venv/bin/python fixtures/generate.py /path/to/models/qwen3.5-4b-heretic

Run it again after changing a case, and commit the result.
"""

import hashlib
import json
import pathlib
import sys

import jinja2
import tokenizers

HERE = pathlib.Path(__file__).resolve().parent

# Every case the tokenizer fixture covers, as (name, text).
TOKENIZER_CASES = [
    ("empty", ""),
    ("ascii", "Hello, world!"),
    ("ascii_sentence", "The quick brown fox jumps over the lazy dog."),
    ("leading_space", "   leading spaces"),
    ("trailing_space", "trailing spaces   "),
    ("only_spaces", "     "),
    ("tabs", "\tcolumn\tseparated\t"),
    ("newlines", "first line\nsecond line\n\nfourth line\n"),
    ("crlf", "windows\r\nline endings\r\n"),
    ("mixed_whitespace", "a \n\t b  \n\n  c"),
    ("digits", "Pi is 3.14159265358979 and 2^16 = 65536"),
    ("cjk", "今日はいい天気です。你好，世界。한국어도 여기에."),
    ("cjk_mixed", "Rust は高速です (fast), 2560 次元"),
    ("emoji", "\U0001f680 ship it \U0001f389\U0001f3fd‍♂️ ❤️\U0001f525"),
    ("emoji_only", "\U0001f602\U0001f602\U0001f602"),
    ("nfc_decomposed", "café vs café"),
    ("control_token_literal", "the literal <|im_start|> in plain text"),
    ("think_literal", "<think> is a tag and </think> closes it"),
    ("both_specials", "<|im_start|>user\nhi<|im_end|>\n<think>\n\n</think>\n\n"),
    ("endoftext_literal", "ends with <|endoftext|>"),
    ("rendered_prompt", "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("long_whitespace_run", "word" + " " * 40 + "word\n\n\n\n" + "\t" * 8 + "end"),
    ("repeated", "ab" * 200),
]

# Every case the chat-template fixture covers. `thinking` of None leaves
# `enable_thinking` undefined, which is how the template is called by default.
CHAT_CASES = [
    ("user_only_thinking", [("user", "What is 2+2?")], True, True),
    ("user_only_no_thinking", [("user", "What is 2+2?")], True, False),
    ("user_only_thinking_undefined", [("user", "What is 2+2?")], True, None),
    ("user_only_no_generation_prompt", [("user", "What is 2+2?")], False, True),
    (
        "system_and_user",
        [("system", "You are terse."), ("user", "Why is the sky blue?")],
        True,
        True,
    ),
    (
        "system_and_user_no_thinking",
        [("system", "You are terse."), ("user", "Why is the sky blue?")],
        True,
        False,
    ),
    (
        "multi_turn",
        [
            ("system", "You are terse."),
            ("user", "First question?"),
            ("assistant", "First answer."),
            ("user", "Second question?"),
        ],
        True,
        True,
    ),
    (
        "multi_turn_no_system",
        [
            ("user", "One?"),
            ("assistant", "Uno."),
            ("user", "Two?"),
            ("assistant", "Dos."),
            ("user", "Three?"),
        ],
        True,
        False,
    ),
    (
        "assistant_with_reasoning_replayed",
        [
            ("user", "First question?"),
            ("assistant", "<think>\nSome reasoning here.\n</think>\n\nFirst answer."),
            ("user", "Second question?"),
        ],
        True,
        True,
    ),
    (
        "trailing_assistant_keeps_reasoning",
        [
            ("user", "First question?"),
            ("assistant", "<think>\nSome reasoning here.\n</think>\n\nFirst answer."),
        ],
        True,
        True,
    ),
    (
        "trailing_assistant_without_reasoning",
        [("user", "First question?"), ("assistant", "Just an answer.")],
        True,
        True,
    ),
    (
        "whitespace_is_trimmed",
        [("system", "  \n padded system \n "), ("user", "\n\n  padded question  \n\n")],
        True,
        True,
    ),
    (
        "non_ascii",
        [
            ("system", "你是一个有用的助手。"),
            ("user", "パリの天気は？ \U0001f324️"),
        ],
        True,
        True,
    ),
    (
        "content_containing_control_tokens",
        [("user", "please echo <|im_start|> and <|im_end|> literally")],
        True,
        True,
    ),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    model_dir = pathlib.Path(sys.argv[1])
    tokenizer_json = model_dir / "tokenizer.json"
    template_jinja = model_dir / "chat_template.jinja"

    tok = tokenizers.Tokenizer.from_file(str(tokenizer_json))
    cases = []
    for name, text in TOKENIZER_CASES:
        with_special = tok.encode(text, add_special_tokens=True).ids
        without_special = tok.encode(text, add_special_tokens=False).ids
        cases.append(
            {
                "name": name,
                "text": text,
                "ids": with_special,
                "ids_no_special": without_special,
                "decoded": tok.decode(with_special, skip_special_tokens=False),
            }
        )
    write(
        "token_ids.json",
        {
            "generated_by": f"python tokenizers {tokenizers.__version__}",
            "tokenizer_json_sha256": sha256(tokenizer_json),
            "vocab_size": tok.get_vocab_size(with_added_tokens=True),
            "cases": cases,
        },
    )

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(template_jinja.read_text())
    cases = []
    for name, turns, add_generation_prompt, thinking in CHAT_CASES:
        messages = [{"role": r, "content": c} for r, c in turns]
        kwargs = {} if thinking is None else {"enable_thinking": thinking}
        cases.append(
            {
                "name": name,
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": thinking,
                "prompt": template.render(
                    messages=messages,
                    add_generation_prompt=add_generation_prompt,
                    **kwargs,
                ),
            }
        )
    write(
        "chat_prompts.json",
        {
            "generated_by": f"jinja2 {jinja2.__version__}",
            "chat_template_sha256": sha256(template_jinja),
            "cases": cases,
        },
    )


def raise_exception(message):
    raise jinja2.TemplateError(message)


def write(name, payload):
    path = HERE / name
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(payload['cases'])} cases)")


if __name__ == "__main__":
    main()
