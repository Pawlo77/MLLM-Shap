"""Shared test helpers for transformers_text connector tests."""


class _TokenizerStub:
    """Minimal tokenizer stub for text connector tests."""

    pad_token_id: int | None = 0
    eos_token_id: int | None = 2
    eos_token: str | None = "</s>"
    pad_token: str | None = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) % 31 for ch in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "|".join(str(i) for i in ids)
