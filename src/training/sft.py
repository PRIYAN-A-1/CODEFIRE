"""Assistant-only loss on approved original instruction JSONL. No external weights."""
import argparse,json,torch
from pathlib import Path
from src.inference.generate import Engine
from src.training.checkpoint import save,rng_state

def format_example(tok,record,context):
    prompt=tok.encode('<|user|>\n'+record['instruction']+'<|end|>\n<|assistant|>\n').ids
    answer=tok.encode(record['response']+'<|end|><|eos|>').ids
    ids=prompt+answer
    if len(ids)>context+1:raise ValueError('Instruction example exceeds context; do not truncate away targets')
    labels=[-100]*len(prompt)+answer
    return ids[:-1],labels[1:]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--tokenizer',default='artifacts/tokenizer.json');p.add_argument('--data',required=True);p.add_argument('--output',default='checkpoints/sft/step-00000001.pt');p.add_argument('--epochs',type=int,default=1);a=p.parse_args()
    e=Engine(a.checkpoint,a.tokenizer,'cuda' if torch.cuda.is_available() else 'cpu');e.model.train();opt=torch.optim.AdamW(e.model.parameters(),lr=1e-5);step=0;target_tokens=0
    for epoch in range(a.epochs):
        for line in open(a.data,encoding='utf-8'):
            r=json.loads(line)
            if r.get('approved') is not True:raise ValueError('Instruction data requires explicit approved=true')
            x,y=format_example(e.tokenizer,r,e.model.config.max_context_length)
            opt.zero_grad();_,loss,_=e.model(torch.tensor([x],device=e.device),torch.tensor([y],device=e.device));loss.backward();torch.nn.utils.clip_grad_norm_(e.model.parameters(),1.);opt.step();step+=1;target_tokens+=sum(label!=-100 for label in y);print(json.dumps({'step':step,'loss':loss.item()}))
    s=e.state;s.update(model=e.model.cpu().state_dict(),optimizer=opt.state_dict(),step=step,rng=rng_state(),stage='sft',tokens=e.state['tokens']+target_tokens,training_config={'sft':True,'epochs':a.epochs,'data':a.data});save(a.output,s)
