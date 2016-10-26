import numpy as np
import re
import gzip

with gzip.open('C://Users/user/feats_gz_file/{April30_2sentence1.mpg}_out_features.gz', 'rb') as f:
    s = f.read()
#f = open('{April23_2sentence1.mpg}_out_features','r+')
#s = f.read()
line = re.split('\n',s)

frame_num = np.zeros([len(line)])
feats = np.zeros([len(line), len(line[0].split('\t'))-1])

for i in range(len(line)):
    current_line = line[i].split('\t')
    for j in range(len(current_line)-1):

        feats[i][j] = float(current_line[j])

np.save('{April30_2sentence1.mpg}_out_features.npy', feats)
