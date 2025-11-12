import tqdm
import faiss
import numpy
import torch
import GPUtil


def get_gpu_num2():
    return torch.cuda.device_count()

def get_gpu_num():
    try:
        ngpu = faiss.get_num_gpus()
    except:
        ngpu = len(GPUtil.getGPUs())
    return ngpu

print(tqdm.__version__)
print(faiss.__version__)
print(numpy.__version__)
print(torch.__version__)
print('GPU Num: ', get_gpu_num())
