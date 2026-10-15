import argparse,json
from pathlib import Path
from safetensors.torch import save_file
from src.training.checkpoint import load
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/latest');p.add_argument('--output',default='artifacts/export');a=p.parse_args();s=load(a.checkpoint);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    save_file({k:v.contiguous().clone() for k,v in s['model'].items()},str(out/'model.safetensors'));(out/'config.json').write_text(json.dumps(s['model_config'],indent=2))
    (out/'model_card.json').write_text(json.dumps({'model':s['model_config'],'training_tokens':s['tokens'],'tokenizer_sha256':s['tokenizer_sha256'],'dataset_sha256':s['manifest_sha256'],'instruction_following':'not_evaluated','status':'research checkpoint'},indent=2))
