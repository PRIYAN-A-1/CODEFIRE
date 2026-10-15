from pathlib import Path
import difflib

def resolve(root,path):
    root=Path(root).resolve();p=(root/path).resolve()
    if not p.is_relative_to(root) or any(part.startswith('.') for part in Path(path).parts):raise ValueError('Path outside workspace or hidden path')
    return p

def read(root,path):
    p=resolve(root,path)
    if p.stat().st_size>100000:raise ValueError('File too large')
    from src.data.collect import SECRET
    text=p.read_text()
    if SECRET.search(text):raise ValueError('Potential secret; refusing to send to model')
    return text

def propose(root,path,replacement):
    original=read(root,path)
    return ''.join(difflib.unified_diff(original.splitlines(True),replacement.splitlines(True),fromfile='a/'+path,tofile='b/'+path))
