import numpy as np
import joblib
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
import os 
import torchaudio.transforms as T
from utils import specAug

SAMPLE_RATE = 16000
CHUNK_LENGTH = 250000
mode = 'fine-tune'

class SpecAugmentation(torch.nn.Module):
    def __init__(self, rate=1.2, time_mask_param = 80, freq_mask_param=80):
        super(SpecAugmentation, self).__init__()

        self.rate = rate
        # self.stretch = T.TimeStretch(n_freq = 512)
        self.TimeMasking = T.TimeMasking(time_mask_param=time_mask_param)
        self.FreMasking = T.FrequencyMasking(freq_mask_param=freq_mask_param)

    def forward(self, x):
        # x = self.stretch(x, self.rate).real

        x = self.TimeMasking(x)

        x = self.FreMasking(x)

        return x

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

# train
df = pd.read_csv('/work/f776655321/DUAL-textless-NER/slue-voxpopuli/slue-voxpopuli_'+ mode + '.tsv',sep='\t')
audio_file_dir = '/work/f776655321/DUAL-textless-NER/slue-voxpopuli/' + mode

output_dir = '/work/f776655321/DUAL-textless-NER/code-data/code/' + mode
extractor = torch.hub.load('s3prl/s3prl', 'hubert_large_ll60k')

modules = list(extractor.model.feature_extractor.conv_layers.children())

modules.append(SpecAugmentation())
extractor.model.feature_extractor.conv_layers = torch.nn.ModuleList(modules)

extractor.eval()


if torch.cuda.is_available():
        extractor = extractor.cuda()

count_prefix = 0
count_latefix = 0
apply_kmeans = ApplyKmeans('/work/f776655321/DUAL-textless-NER/speeech_content_encoder/km_100h_c128/km_feat_layer_22')

for file in tqdm(df['id'].values, desc='transforming lxt data to discrete code'):
    
    output_file = f"context-{count_prefix}_{count_latefix}-SpecAug"
    audio_file = os.path.join(audio_file_dir, file+'.ogg')
    
    #specaug wavform
    # wavs = specAug(audio_file,SAMPLE_RATE)
    wavs = reader(audio_file)

    wavs = wavs.cuda()

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

        code = apply_kmeans(feature.cuda())

        
        
    else:
        feature = extractor([wavs])

        feature = feature['hidden_state_22']
    
        
        code = apply_kmeans(feature.squeeze().cuda())

    code = torch.tensor(code)

    merged_code, counts = torch.unique_consecutive(code, return_counts=True)

    np.savetxt(os.path.join(output_dir, output_file+'.code'), merged_code.long(), fmt='%i')    
    np.savetxt(os.path.join(output_dir, output_file+'.cnt'), counts.long(), fmt='%i')

    if(count_latefix < 53):
        count_latefix += 1
    else:
        count_prefix += 1
        count_latefix = 0