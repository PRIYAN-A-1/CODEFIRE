import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.system.hardware import report
print(json.dumps(report(),indent=2))
