import torch,pytest
from src.model.transformer import ModelConfig,Transformer
@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(2)
@pytest.fixture
def model():
    torch.manual_seed(7)
    return Transformer(ModelConfig(vocab_size=512,hidden_size=32,num_layers=2,num_attention_heads=4,intermediate_size=64,max_context_length=128)).eval()
@pytest.fixture
def corpus(tmp_path):
    from src.data.collect import collect
    from src.tokenizer.train_tokenizer import train
    from src.data.prepare import prepare
    sources=[]
    texts={'train':'def summation(values):\n    total = 0\n    for value in values:\n        total += value\n    return total\n','validation':'def reverse_words(sentence):\n    words = sentence.split()\n    words.reverse()\n    return " ".join(words)\n','test':'def enumerate_results(elements):\n    lookup = {}\n    for position, element in enumerate(elements):\n        lookup[element] = position\n    return lookup\n'}
    for split,text in texts.items():
        path=tmp_path/split;path.mkdir();(path/'code.py').write_text(text);sources.append({'path':str(path),'repository':split,'split':split,'approved':True,'license_spdx':'CC0-1.0'})
    c={'sources':sources,'records':str(tmp_path/'records.jsonl'),'tokenizer':str(tmp_path/'tokenizer.json'),'output':str(tmp_path/'tokens'),'vocab_size':512,'shard_tokens':64}
    collect(c);train(c['records'],c['tokenizer'],512);prepare(c);return c
