import json, shutil, os
import psutil, torch

def report(path='.'):
    disk=shutil.disk_usage(path);cuda=torch.cuda.is_available()
    gpu={'available':cuda,'name':None,'vram_bytes':None,'bf16':False}
    if cuda:gpu.update(name=torch.cuda.get_device_name(0),vram_bytes=torch.cuda.get_device_properties(0).total_memory,bf16=torch.cuda.is_bf16_supported())
    return {'gpu':gpu,'torch':torch.__version__,'cuda_version':torch.version.cuda,'cpu_cores':os.cpu_count(),'ram_available_bytes':psutil.virtual_memory().available,'disk_free_bytes':disk.free,'recommended_precision':'bf16' if gpu['bf16'] else ('fp16' if cuda else 'fp32'),'batch_size':'unmeasured; use probe','recommended_model':'23M first; probe before 134M'}
if __name__=='__main__':print(json.dumps(report(),indent=2))
