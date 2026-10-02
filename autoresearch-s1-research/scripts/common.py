"""Shared, stdlib-only validation primitives. Never imports task code."""
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone


def load(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate JSON key: ' + key)
            out[key] = value
        return out
    def bad(value):
        raise ValueError('nonfinite JSON number: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=pairs, parse_constant=bad)


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label + ' must be a finite number')
    return float(value)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, relative, existing=True):
    root = Path(root).resolve()
    raw = Path(relative)
    if raw.is_absolute() or '..' in raw.parts or not raw.parts:
        raise ValueError('unsafe relative path: ' + str(relative))
    current = root
    for part in raw.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('symlink disallowed: ' + str(relative))
    result = current.resolve()
    if root != result and root not in result.parents:
        raise ValueError('path escapes root')
    if existing and not result.is_file():
        raise ValueError('missing regular file: ' + str(relative))
    return result


def evidence(root, item):
    if not isinstance(item, dict) or not isinstance(item.get('sha256'), str):
        raise ValueError('evidence needs path and sha256')
    path = inside(root, item['path'])
    if digest(path) != item['sha256']:
        raise ValueError('evidence hash mismatch: ' + item['path'])
    return path


def atomic_json(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('refuse symlink output')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def utc():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('event time must include timezone')
    return parsed.timestamp()


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError('interval must have positive duration')
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def duration(active, excluded=()):
    good, bad = merged(active), merged(excluded)
    total = sum(b - a for a, b in good)
    for a, b in good:
        for c, d in bad:
            total -= max(0, min(b, d) - max(a, c))
    return total
