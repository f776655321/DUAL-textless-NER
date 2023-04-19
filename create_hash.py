import pandas as pd
from tqdm import tqdm
import numpy as np
import joblib
import json
df = pd.read_csv('/tmp2/b09902111/slue-voxpopuli/slue-voxpopuli_dev.tsv',sep='\t')

hashed = {}
count_prefix = 0
count_latefix = 0

for file in tqdm(df['id'].values, desc='hashing file name to context_id'):

    output_hash = f"context-{count_prefix}_{count_latefix}"
    hashed[file] = output_hash

    if(count_latefix < 53):
        count_latefix += 1
    else:
        count_prefix += 1
        count_latefix = 0

with open("dev-hash2context.json", "w") as file:
    json.dump(hashed, file)


