import subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if __name__=='__main__':
    for module in ['src.data.collect','src.tokenizer.train_tokenizer','src.data.prepare']:
        subprocess.run([sys.executable,'-m',module,'--config','configs/dataset.yaml'],check=True)
    command=[sys.executable,'-m','src.training.train','--config','configs/train_smoke.yaml']
    if Path('checkpoints/latest').exists():command+=['--resume','checkpoints/latest']
    subprocess.run(command,check=True)
    subprocess.run([sys.executable,'-m','src.evaluation.evaluate'],check=True)
