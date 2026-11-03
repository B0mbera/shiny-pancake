"""Pancake-chan, the project's mascot, plus tiny terminal styling helpers.

The art is plain text so it works in any terminal; colours are only added when the
output stream is a TTY and ``NO_COLOR`` is not set (https://no-color.org).
"""

from __future__ import annotations

import os
import sys
from typing import TextIO

from . import __version__

# mood -> (unicode face, ascii fallback)
_FACES = {
    "happy": (("◕", "ᴗ", "◕"), ("o", "u", "o")),
    "working": (("•", "ω", "•"), ("*", "w", "*")),
    "proud": (("^", "ᴗ", "^"), ("^", "u", "^")),
    "oops": (("×", "﹏", "×"), ("x", "n", "x")),
}

_TEMPLATE = """\
                   _______
                  [_______]
          .-~~~~~~~~~~~~~~~~~~~~~-.
        .'                         '.
       (      {l}       {m}       {r}      )
        '.   {b}               {b}   .'
       (__'-.._________________..-'__)
      (_________________________________)
       (_______________________________)"""

TAGLINE = "a stack of LLM agents, served warm"


def _supports(stream: TextIO, text: str) -> bool:
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def use_color(stream: TextIO | None = None) -> bool:
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") or os.environ.get("SP_NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


class Style:
    """ANSI helpers that degrade to plain text."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, t: str) -> str:
        return self._wrap("1", t)

    def dim(self, t: str) -> str:
        return self._wrap("2", t)

    def amber(self, t: str) -> str:
        return self._wrap("38;5;214", t)

    def butter(self, t: str) -> str:
        return self._wrap("38;5;228", t)

    def pink(self, t: str) -> str:
        return self._wrap("38;5;211", t)

    def green(self, t: str) -> str:
        return self._wrap("38;5;114", t)

    def red(self, t: str) -> str:
        return self._wrap("38;5;203", t)

    def blue(self, t: str) -> str:
        return self._wrap("38;5;111", t)


def mascot(mood: str = "happy", *, stream: TextIO | None = None, color: bool | None = None) -> str:
    """Return Pancake-chan as a multi-line string."""
    stream = stream or sys.stdout
    unicode_face, ascii_face = _FACES.get(mood, _FACES["happy"])
    face, blush = unicode_face, "˶˶"
    if not _supports(stream, "".join(face) + blush):
        face, blush = ascii_face, "''"
    art = _TEMPLATE.format(l=face[0], m=face[1], r=face[2], b=blush)
    style = Style(use_color(stream) if color is None else color)
    if not style.enabled:
        return art
    lines = art.splitlines()
    out = [style.butter(lines[0]), style.butter(lines[1])]
    for line in lines[2:]:
        out.append(style.amber(line))
    # recolour the face line: blush stays pink, eyes bold
    out[4] = (
        style.amber("       (      ")
        + style.bold(face[0])
        + "       "
        + style.pink(face[1])
        + "       "
        + style.bold(face[2])
        + style.amber("      )")
    )
    out[5] = (
        style.amber("        '.   ")
        + style.pink(blush)
        + "               "
        + style.pink(blush)
        + style.amber("   .'")
    )
    return "\n".join(out)


def banner(subtitle: str = "", *, mood: str = "happy", stream: TextIO | None = None) -> str:
    stream = stream or sys.stdout
    style = Style(use_color(stream))
    title = style.bold(f"shiny-pancake v{__version__}")
    tag = style.dim(TAGLINE)
    lines = [mascot(mood, stream=stream), "", f"          {title}  ·  {tag}"]
    if subtitle:
        lines.append(f"          {style.blue(subtitle)}")
    return "\n".join(lines) + "\n"


def print_banner(subtitle: str = "", *, mood: str = "happy", stream: TextIO | None = None) -> None:
    stream = stream or sys.stdout
    if os.environ.get("SP_NO_ART"):
        return
    print(banner(subtitle, mood=mood, stream=stream), file=stream)
