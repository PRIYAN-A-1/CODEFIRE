"""Optional Docker execution only; never falls back to host execution."""
import shutil,subprocess,tempfile
from pathlib import Path

def test_python(code,test_code,image='python:3.12-slim'):
    if not shutil.which('docker'):raise RuntimeError('Docker required. No host execution fallback.')
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp);(path/'candidate.py').write_text(code+'\n'+test_code)
        cmd=['docker','run','--rm','--pull=never','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=64','--memory=256m','--cpus=.5','--user=65534:65534','--tmpfs=/tmp:rw,noexec,nosuid,size=16m','-v',str(path.resolve())+':/work:ro',image,'python','-I','-B','/work/candidate.py']
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
        return {'returncode':result.returncode,'stdout':result.stdout[:10000],'stderr':result.stderr[:10000]}
