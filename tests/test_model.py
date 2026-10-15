import pytest,torch,yaml
from src.model.transformer import ModelConfig,Transformer,rope

def test_shape_and_gradients(model):
    x=torch.randint(512,(2,10));logits,loss,_=model(x,x);assert logits.shape==(2,10,512);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
def test_causality(model):
    a=torch.randint(512,(1,12));b=a.clone();b[:,7:]=torch.randint(512,(1,5));torch.testing.assert_close(model(a)[0][:,:7],model(b)[0][:,:7])
@pytest.mark.parametrize('chunk',[1,3])
def test_cache_equivalence(model,chunk):
    x=torch.randint(512,(1,11));full=model(x)[0];first,_,cache=model(x[:,:5],use_cache=True);outputs=[first]
    for pos in range(5,11,chunk):
        logits,_,cache=model(x[:,pos:pos+chunk],cache=cache,use_cache=True);outputs.append(logits)
    torch.testing.assert_close(torch.cat(outputs,1),full,atol=2e-6,rtol=2e-5)
def test_rope_preserves_norm():
    x=torch.randn(2,4,7,8);y=rope(x,11,10000);torch.testing.assert_close(x.square().sum(-1),y.square().sum(-1))
def test_context_rejection(model):
    with pytest.raises(ValueError):model(torch.zeros(1,129,dtype=torch.long))
def test_invalid_config():
    with pytest.raises(AssertionError):ModelConfig(hidden_size=31,num_attention_heads=4)
@pytest.mark.parametrize('preset',['23m','134m'])
def test_presets_forward_backward(preset):
    m=Transformer(ModelConfig(**yaml.safe_load(open(f'configs/model_{preset}.yaml'))));x=torch.tensor([[1,2,3]]);_,loss,_=m(x,x);loss.backward();assert torch.isfinite(loss)
    d=m.config.hidden_size;n=m.config.num_layers;f=m.config.intermediate_size;v=m.config.vocab_size
    assert m.parameter_count()==v*d*(1 if m.config.tie_word_embeddings else 2)+n*(4*d*d+3*d*f+2*d)+d
def test_gradient_checkpointing_matches(model):
    import copy
    other=copy.deepcopy(model);model.train();other.train();other.gradient_checkpointing=True;x=torch.randint(512,(1,12));model(x,x)[1].backward();other(x,x)[1].backward()
    for a,b in zip(model.parameters(),other.parameters()):torch.testing.assert_close(a.grad,b.grad)
def test_tiny_batch_overfit(model):
    model.train();x=torch.tensor([[1,5,7,8,2]]);y=torch.tensor([[5,7,8,2,1]]);opt=torch.optim.AdamW(model.parameters(),lr=.01);initial=model(x,y)[1].item()
    for _ in range(30):opt.zero_grad();loss=model(x,y)[1];loss.backward();opt.step()
    assert model(x,y)[1].item()<initial*.25
