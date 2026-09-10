#!/usr/bin/env python3
"""Export this pinned runtime privately, preserving an application's public b12x ABI.

Only package identifiers, qualified package/operator strings, and the compile
cache environment key change. Kernel math, signatures, and policies stay intact.
The manifest binds each generated file to its original source bytes.
"""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import tokenize

PACKAGE = 'trellismx_b12x'

def rewrite_source(source):
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        value = token.string
        if token.type == tokenize.NAME and value == 'b12x':
            value = PACKAGE
        elif token.type == tokenize.STRING:
            try:
                literal = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                literal = None
            if isinstance(literal, str):
                if re.fullmatch(r'b12x(?:(?:\.|::)[\w.:]*)?', literal):
                    value = repr(PACKAGE + literal[4:])
                elif literal == 'B12X_COMPILE_CACHE_DIR':
                    value = repr('TRELLISMX_COMPILE_CACHE_DIR')
        tokens.append(token._replace(string=value))
    result = tokenize.untokenize(tokens)
    ast.parse(result)
    return result

def export(source, destination):
    package = destination/PACKAGE
    if package.exists():
        raise FileExistsError(package)
    manifest = {'namespace': PACKAGE, 'format_version': 1, 'files': {}}
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source)
        if not path.is_file() or '__pycache__' in relative.parts or path.suffix in ('.pyc', '.so'):
            continue
        original = path.read_bytes()
        generated = rewrite_source(original.decode()).encode() if path.suffix == '.py' else original
        output = package/relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(generated)
        manifest['files'][relative.as_posix()] = {
            'source_sha256': hashlib.sha256(original).hexdigest(),
            'generated_sha256': hashlib.sha256(generated).hexdigest(),
        }
    (destination/'trellismx-runtime-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps({'files': len(export(args.source, args.destination)['files'])}))
