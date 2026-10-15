import argparse,hashlib
from pathlib import Path
import torch
from tokenizers import Tokenizer
from src.model.transformer import ModelConfig,Transformer
from src.training.checkpoint import load
class Engine:
    def __init__(self,checkpoint,tokenizer,device='cpu'):
        s=load(checkpoint)
        if hashlib.sha256(Path(tokenizer).read_bytes()).hexdigest()!=s['tokenizer_sha256']:raise ValueError('Tokenizer checksum mismatch')
        self.tokenizer=Tokenizer.from_file(str(tokenizer));self.model=Transformer(ModelConfig(**s['model_config'])).to(device);self.model.load_state_dict(s['model']);self.model.eval();self.device=device;self.state=s
    @torch.inference_mode()
    def stream(self,prompt,max_new_tokens=128,temperature=.7,top_k=40,top_p=.95,seed=42,repetition_penalty=1.0):
        ids=self.tokenizer.encode(prompt).ids or [self.tokenizer.token_to_id('<|bos|>')]
        c=self.model.config
        if len(ids)+max_new_tokens>c.max_context_length:raise ValueError(f'Prompt plus output exceeds context {c.max_context_length}')
        x=torch.tensor([ids],device=self.device);cache=None;gen=torch.Generator(device=self.device).manual_seed(seed)
        generated=[];previous=''
        for _ in range(max_new_tokens):
            logits,_,cache=self.model(x,cache=cache,use_cache=True);scores=logits[0,-1].float().clone()
            scores[self.tokenizer.get_vocab_size():]=-float('inf')
            for i in set(ids+generated):scores[i]=scores[i]/repetition_penalty if scores[i]>0 else scores[i]*repetition_penalty
            if temperature==0:next_id=int(scores.argmax())
            else:
                scores/=temperature
                if top_k>0:scores[scores<torch.topk(scores,min(top_k,len(scores))).values[-1]]=-float('inf')
                ordered,indices=scores.sort(descending=True);cumulative=ordered.softmax(-1).cumsum(-1);remove=cumulative>top_p;remove[1:]=remove[:-1].clone();remove[0]=False;scores[indices[remove]]=-float('inf')
                next_id=int(torch.multinomial(scores.softmax(-1),1,generator=gen))
            if next_id==self.tokenizer.token_to_id('<|eos|>'):break
            generated.append(next_id);text=self.tokenizer.decode(generated)
            # Buffer incomplete byte sequences until the decoder has complete Unicode.
            if not text.endswith('\ufffd'):
                yield text[len(previous):];previous=text
            x=torch.tensor([[next_id]],device=self.device)
        text=self.tokenizer.decode(generated)
        if len(text)>len(previous):yield text[len(previous):]
    def generate(self,prompt,**kwargs):return ''.join(self.stream(prompt,**kwargs))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/latest');p.add_argument('--tokenizer',default='artifacts/tokenizer.json');p.add_argument('--prompt',required=True);p.add_argument('--max-new-tokens',type=int,default=64);a=p.parse_args();e=Engine(a.checkpoint,a.tokenizer);print(e.generate(a.prompt,max_new_tokens=a.max_new_tokens))
