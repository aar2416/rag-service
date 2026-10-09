import re

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        if len(piece) > size:
            pieces.extend(_split(piece, size, rest))
        elif piece:
            pieces.append(piece)
    return pieces


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Greedy recursive splitter: paragraphs -> lines -> sentences -> words, packed up to `size`
    characters, with roughly `overlap` characters carried over between neighbouring chunks."""
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    overlap = max(0, min(overlap, size // 2))

    chunks: list[str] = []
    current = ""
    for piece in _split(text, size, _SEPARATORS):
        if current and len(current) + len(piece) > size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            # start the overlap on a word boundary
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            current = tail
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks
