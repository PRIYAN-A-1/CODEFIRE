import shutil
from pathlib import Path

def require_space(path, needed, reserve=64*1024*1024):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(path).free < needed+reserve:raise OSError('Insufficient free disk: relocate output or free storage; no source files were deleted')
