"""Incremental PII redactor for token streams.

The problem: PII arrives split across deltas (``"john.d"``, ``"oe@exa"``,
``"mple.com"``). Redacting each delta alone misses it, and buffering the whole
response ruins time-to-first-token.

The approach: hold back only the *suffix that could still turn into PII*.

* Every pattern we redact is a contiguous run of characters from a small
  alphabet: emails use ``[A-Za-z0-9_.%+@-]``, while SSNs and card numbers are digits
  with single space/dash separators. So text before the start of the trailing
  run in that alphabet can never become part of an incomplete match, and it is
  safe to emit after redaction.
* In practice the hold-back is the last partial word (or the trailing digit
  group), a handful of characters. Ordinary prose streams out with roughly
  one-token delay.
* The hold-back is capped at ``MAX_HOLD`` characters (>= the longest possible
  match), so memory per stream is O(MAX_HOLD + chunk) however long the response.

Invariant (property-tested): for any way of splitting a string into chunks,
``"".join(feed(c) for c in chunks) + flush() == redact_text(full_string)``.
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

# Boundaries are ASCII-only on purpose: every lookbehind char class must sit
# inside the hold-back alphabet below, so a held suffix never loses the context
# that decides whether it matches.
_EMAIL = r"(?<![A-Za-z0-9_.%+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){0,8}\.[A-Za-z]{2,24}(?![A-Za-z0-9_-])"
# separator must be consistent: 123-45-6789 or 123 45 6789
_SSN = r"(?<![\d-])\d{3}(?P<sep>[- ])\d{2}(?P=sep)\d{4}(?![\d-])"
# 13-19 digits, optional single space/dash between digits; Luhn-checked below
_CARD = r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"

PII_RE = re.compile(rf"(?P<email>{_EMAIL})|(?P<ssn>{_SSN})|(?P<card>{_CARD})", re.ASCII)

# A suffix that might be the beginning of a match that is still being streamed.
_TAIL_RE = re.compile(r"(?:[A-Za-z0-9_.%+@-]+|(?:\d[ -]?)+)$", re.ASCII)
# Superset of every lookbehind class used above.
_TOKEN_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.%+@-")

# Longest possible match (email: 64 + 1 + 253 + slack). Bounds the hold-back.
MAX_HOLD = 336


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _is_pii(m: re.Match) -> bool:
    if m.lastgroup == "card":
        digits = re.sub(r"[ -]", "", m.group())
        return 13 <= len(digits) <= 19 and _luhn_ok(digits)
    return True


def _redact_upto(text: str, end: int) -> tuple[str, int]:
    """Redact PII in ``text[:end]``. Returns (redacted_text, actual_end).

    ``actual_end`` may move left of ``end`` if a match straddles the boundary,
    so a match is never emitted half-redacted.
    """
    out: list[str] = []
    pos = 0
    for m in PII_RE.finditer(text):
        if m.start() >= end:
            break
        if m.end() > end:  # straddles the cut: hold the whole match back
            end = m.start()
            break
        if _is_pii(m):
            out.append(text[pos : m.start()])
            out.append(REDACTED)
            pos = m.end()
    out.append(text[pos:end])
    return "".join(out), end


def redact_text(text: str) -> str:
    return _redact_upto(text, len(text))[0]


class StreamRedactor:
    """Feed deltas in, get safe-to-emit (already redacted) text out."""

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = ""

    @property
    def pending(self) -> int:
        return len(self._buf)

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        buf = self._buf + delta

        # Only scan the last MAX_HOLD chars for the risky suffix (keeps it O(chunk)).
        window_start = max(0, len(buf) - MAX_HOLD)
        tail = _TAIL_RE.search(buf, window_start)
        cut = tail.start() if tail else len(buf)
        # A digit run can start mid-token ("x-123-45-6789"). Back up to the token
        # start, so the held text keeps the left context the lookbehinds check.
        while cut > window_start and buf[cut - 1] in _TOKEN_CHARS:
            cut -= 1

        emitted, cut = _redact_upto(buf, cut)
        self._buf = buf[cut:]
        return emitted

    def flush(self) -> str:
        out = redact_text(self._buf)
        self._buf = ""
        return out
