import argparse,ast,json,math,time
from pathlib import Path
import torch
from src.inference.generate import Engine
from src.data.dataset import TokenDataset

def evaluate(checkpoint,tokenizer,dataset,split='test',batches=10):
    e=Engine(checkpoint,tokenizer);seq=min(32,e.model.config.max_context_length);ds=TokenDataset(dataset,split,seq)
    g=torch.Generator().manual_seed(43);losses=[];tic=time.perf_counter()
    with torch.no_grad():
        for _ in range(batches):
            x,y=ds.batch(1,g,'cpu');losses.append(e.model(x,y)[1].item())
    elapsed=time.perf_counter()-tic
    prompts=['def unique_sorted(values):\n','def is_even(number):\n'];failures=[];valid=0
    for prompt in prompts:
        result=e.generate(prompt,max_new_tokens=min(24,e.model.config.max_context_length-len(e.tokenizer.encode(prompt).ids)),temperature=0)
        try:ast.parse(prompt+result);valid+=1
        except SyntaxError:failures.append({'prompt':prompt,'completion':result,'reason':'Python syntax error'})
    mean=sum(losses)/len(losses)
    report={'model':e.model.config.name,'parameters':e.model.parameter_count(),'checkpoint':str(checkpoint),'training_tokens':e.state['tokens'],'split':split,'validation_loss':mean if split=='validation' else None,'test_loss':mean if split=='test' else None,'perplexity':math.exp(min(mean,80)),'tokens_per_second':seq*batches/elapsed,'syntax_validity':valid/len(prompts),'syntax_tasks':len(prompts),'unit_test_pass_rate':None,'pass_at_1':None,'compile_success':None,'bug_fix_success_rate':None,'status':'fixture_evaluated_not_a_benchmark','failures':failures}
    Path('evaluations').mkdir(exist_ok=True);Path('evaluations/latest.json').write_text(json.dumps(report,indent=2));Path('evaluations/latest.md').write_text('# Fixture evaluation\n\nThese are tiny fixture measurements, not EvalPlus/LiveCodeBench results.\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n');return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/latest');p.add_argument('--tokenizer',default='artifacts/tokenizer.json');p.add_argument('--dataset',default='data/tokenized');p.add_argument('--split',choices=['test','validation'],default='test');a=p.parse_args();torch.set_num_threads(2);print(json.dumps(evaluate(a.checkpoint,a.tokenizer,a.dataset,a.split),indent=2))
