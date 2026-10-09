"""Authentic preset quote and simple demonstration tokens, not a trained tokenizer."""

from dataclasses import replace
import re
from .architecture import ModelConfig

QUOTES = {
    "zh": "机器常常令我惊讶。",
    "en": "Machines take me by surprise with great frequency.",
}
QUOTE = QUOTES["zh"]
QUOTE_SOURCE = "https://www.cse.msu.edu/~cse841/papers/Turing.html"


def quote_tokens(language="zh"):
    quote = QUOTES[language]
    return (
        tuple(quote)
        if language == "zh"
        else tuple(re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|[^\s]", quote))
    )


def join_tokens(tokens, language="zh"):
    if language == "zh":
        return "".join(tokens)
    return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(tokens))


def demo_config(blocks=("dense", "moe"), language="zh"):
    tokens = quote_tokens(language)
    extras = (
        tuple("智能学习思考未来") if language == "zh" else ("AI", "thought", "future")
    )
    vocab = tuple(dict.fromkeys(tokens + extras))
    ids = tuple(vocab.index(token) for token in tokens)
    return replace(ModelConfig(), vocab=vocab, context=ids, blocks=blocks, heads=4)
