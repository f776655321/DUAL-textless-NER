import numpy as np
import joblib
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
import os 
from transformers import AutoModel
from datasets import load_dataset, Dataset, load_from_disk
from collections import defaultdict

SAMPLE_RATE = 16000
CHUNK_LENGTH = 250000
# mode = 'dev'

class ApplyKmeans(object):
    def __init__(self, km_path, return_diff=False):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.return_diff = return_diff
        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

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

for mode in ["train", "validation", "test"]:
    ds = load_from_disk(f"slue-sa/{mode}.hf")
    output_dir = 'code-data-sa/code/' + mode
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    extractor = torch.hub.load('s3prl/s3prl', 'hubert_large_ll60k')
    # extractor = AutoModel.from_pretrained("facebook/hubert-large-ll60k", output_hidden_states = True)
    extractor.eval()

    if torch.cuda.is_available():
        extractor = extractor.cuda()

    titles = defaultdict(int)
    apply_kmeans = ApplyKmeans('speeech-content-encoder/km_100h_c128/km_feat_layer_22')
    # apply_kmeans = ApplyKmeans('/home/yuxiang1234/DUAL-textless-NER/slue-sqa-model/L22500.bin')

    for line in tqdm(ds):
        id = line["id"]
        speaker_id = line["speaker_id"]
        
        wavs = torch.FloatTensor(line["audio"]["array"])
        wavs = wavs.cuda()
        output_file = f"{id }_{speaker_id}"
        # try:

        if len(wavs) > 20 * SAMPLE_RATE:
            print(f'{id }-{speaker_id} is too long')
            chunks = torch.split(wavs, CHUNK_LENGTH)
            for i, chunk in enumerate(chunks): 
                feat = extractor([chunk])
                feat = feat['hidden_state_22'].squeeze()
                
                # feat = extractor(chunk.unsqueeze(0))
                # feat = feat['hidden_states'][22].squeeze()
                
                if i == 0:
                    feature = feat
                else: 
                    feature = torch.cat([feature, feat], dim = 0)
            code = apply_kmeans(feature.cuda())
            
            
        else:
            feature = extractor([wavs])
            # feature = extractor(wavs.unsqueeze(0))

            feature = feature['hidden_state_22']
            # feature = feature['hidden_states'][22]

            code = apply_kmeans(feature.squeeze().cuda())

        code = torch.tensor(code)

        merged_code, counts = torch.unique_consecutive(code, return_counts=True)

        np.savetxt(os.path.join(output_dir, output_file+'.code'), merged_code.long(), fmt='%i')    
        np.savetxt(os.path.join(output_dir, output_file+'.cnt'), counts.long(), fmt='%i')
        # except:
        #     print("error")
