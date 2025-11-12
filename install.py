import tqdm
import faiss
import numpy
import torch
import GPUtil


def get_gpu_num2():
    return torch.cuda.device_count()

def get_gpu_num():
    try:
        ngpu = torch.cuda.device_count()
        
    except:
        ngpu = len(GPUtil.getGPUs())
    return ngpu

print(tqdm.__version__)
print(faiss.__version__)
print(numpy.__version__)
print(torch.__version__)
print('GPU Num: ', get_gpu_num())


import numpy as np
data = np.load('data/knns/part1_test/faiss_k_80.npz')
# data = np.load('faiss_k_80.npz')
print(data.files)  # 파일 안에 저장된 배열 이름 리스트 출력

nbrs = data['nbrs'] if 'nbrs' in data else None
dists = data['dists'] if 'dists' in data else None

print('nbrs shape:', nbrs.shape if nbrs is not None else 'None')
print('dists shape:', dists.shape if dists is not None else 'None')

arr = data['data']
print(arr.shape)
print(arr.dtype)

nbrs = arr[:, 0, :].astype(int)     # 이웃 인덱스 (int로 변환 필요)
dists = arr[:, 1, :]                # 거리 (float64)

print('nbrs shape:', nbrs.shape)   # (584013, 80)
print('dists shape:', dists.shape) # (584013, 80)