"""Compose Code Edit source blocks into a browser document after patching.

This is a review candidate, not installed in the running training service.
"""
import re

class MissingDocument(ValueError):
    """The dataset source has no HTML document to render."""


def compose_document(source):
    text = source.strip()
    # The current patch applicator can strip the first and last fences from a
    # compound HTML/CSS document. Recover only that known block representation.
    if not text.startswith('```') and re.search(r'```\s*```(?:css|javascript)\s*\n', text):
        text = '```html\n' + text + '\n```'
    if not text.startswith('```'):
        return source
    pattern = re.compile(r'```(html|css|javascript)\s*\n([\s\S]*?)```')
    matches = list(pattern.finditer(text))
    cursor = 0
    blocks = []
    for match in matches:
        if text[cursor:match.start()].strip():
            raise ValueError('Unexpected text between source blocks')
        blocks.append((match[1], match[2]))
        cursor = match.end()
    if text[cursor:].strip() or not blocks:
        raise ValueError('Incomplete or unsupported source blocks')
    documents = [body for language, body in blocks if language == 'html']
    if len(documents) != 1:
        raise MissingDocument('Exactly one HTML document is required')
    document = documents[0]
    css = '\n'.join(body for language, body in blocks if language == 'css')
    js = '\n'.join(body for language, body in blocks if language == 'javascript')
    if css:
        if re.search(r'</style\s*', css, re.I):
            raise ValueError('CSS block contains an HTML style terminator')
        style = '<style>\n' + css + '\n</style>\n'
        match = re.search(r'</head\s*>', document, re.I)
        if match:
            document = document[:match.start()] + style + document[match.start():]
        else:
            document = style + document
    if js:
        if re.search(r'</script\s*', js, re.I):
            raise ValueError('JavaScript block contains an HTML script terminator')
        script = '<script>\n' + js + '\n</script>\n'
        match = re.search(r'</body\s*>', document, re.I)
        if match:
            document = document[:match.start()] + script + document[match.start():]
        else:
            document += script
    return document
