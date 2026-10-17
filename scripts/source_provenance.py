"""Check original experiment sources and their documented working copies."""
import ast
import hashlib
from pathlib import Path
import zipfile

ARCHIVE = 'reproduction/frozen_source.zip'


def executable_tree(source):
    tree = ast.parse(source.decode('utf-8-sig'))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def verify_source(root, relative, expected):
    root = Path(root).resolve()
    relative = relative.replace('\\', '/')
    path = (root / relative).resolve()
    path.relative_to(root)
    archive = root / ARCHIVE
    if archive.is_file():
        with zipfile.ZipFile(archive) as snapshot:
            original = snapshot.read(relative)
    else:
        original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != expected:
        raise ValueError('Original frozen source changed: ' + relative)
    if path.suffix == '.py':
        current = path.read_bytes()
        if current != original and executable_tree(current) != executable_tree(original):
            raise ValueError('Executable source differs from frozen source: ' + relative)
    elif path.suffix != '.md' and path.read_bytes() != original:
        raise ValueError('Frozen input differs from original: ' + relative)
    return original
