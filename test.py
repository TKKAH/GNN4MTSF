import os

import numpy as np

test_cache_file = os.path.join('dataset/m4', 'test.npz')
values = np.load(test_cache_file, allow_pickle=True)
print(values.shape)
ll=[]
for i in values:
    ll.append(len(i))
print(ll)