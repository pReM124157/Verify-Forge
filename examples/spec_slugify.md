# Slugify

Module: slugify

## Requirements
- Provide `slugify(text: str) -> str`.
- Output is lowercase ASCII containing only letters, digits and single hyphens.
- Whitespace and punctuation runs collapse to a single hyphen.
- Leading and trailing hyphens are stripped.
- Accented characters are transliterated (e.g. "Café" -> "cafe").
- Empty or all-punctuation input returns an empty string.
