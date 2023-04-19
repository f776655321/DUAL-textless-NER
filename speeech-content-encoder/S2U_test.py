import numpy as np
import joblib
import torch
import torchaudio 
import pandas as pd
from tqdm import tqdm
import os 

SAMPLE_RATE = 16000
CHUNK_LENGTH = 250000
class ApplyKmeans(object):
    def __init__(self, km_path, return_diff=False):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.return_diff = return_diff
        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda(6)
            self.Cnorm = self.Cnorm.cuda(6)

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = torch.sqrt(
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            min_dist = dist.detach().min(dim=1)
            if self.return_diff:
                return min_dist.indices.cpu().numpy(), min_dist.values.cpu().numpy()
            else:
                return min_dist.indices.cpu().numpy()
        else:
            dist = np.sqrt(
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            if self.return_diff:
                return np.argmin(dist, axis=1), np.min(dist, axis=1)
            else:
                return np.argmin(dist, axis=1)

def reader(fname):
    wav, ori_sr = torchaudio.load(fname)
    if ori_sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(ori_sr, SAMPLE_RATE)(wav)
    return wav.squeeze()


# train
df = pd.read_csv('/tmp2/b09902111/slue-voxpopuli/slue-voxpopuli_fine-tune.tsv',sep='\t')
audio_file_dir = '/tmp2/b09902111/slue-voxpopuli/fine-tune'

output_dir = '/tmp2/b09902111/data/fine-tune'
extractor = torch.hub.load('s3prl/s3prl', 'hubert_large_ll60k')
extractor.eval()

if torch.cuda.is_available():
        extractor = extractor.cuda(6)

count_prefix = 0
count_latefix = 0
apply_kmeans = ApplyKmeans('/tmp2/b09902111/DUAL-textless-SQA/speeech-content-encoder/km_100h_c128/km_feat_layer_22')

for file in tqdm(df['id'].values, desc='transforming lxt data to discrete code'):
    output_file = f"context-{count_prefix}_{count_latefix}"
    audio_file = os.path.join(audio_file_dir, file+'.ogg')
    wavs = reader(audio_file)
    wavs = wavs.cuda(6)  

    if len(wavs) > 20 * SAMPLE_RATE: 
        print(f'{file} too long')
        chunks = torch.split(wavs, CHUNK_LENGTH) 
        for i, chunk in enumerate(chunks): 
            feat = extractor([chunk])
            feat = feat['hidden_state_22'].squeeze()
            
            if i == 0:
                feature = feat
            else: 
                feature = torch.cat([feature, feat], dim = 0)

        code = apply_kmeans(feature.cuda(6)) 
    else:
        feature = extractor([wavs])    

        
        code = apply_kmeans(feature['hidden_state_22'].squeeze().cuda(6))

    code = torch.tensor(code)

    merged_code, counts = torch.unique_consecutive(code, return_counts=True)
    np.savetxt(os.path.join(output_dir, output_file+'.code'), merged_code.long(), fmt='%i')    
    np.savetxt(os.path.join(output_dir, output_file+'.cnt'), counts.long(), fmt='%i')
    if(count_latefix < 53):
        count_latefix += 1
    else:
        count_prefix += 1
        count_latefix = 0