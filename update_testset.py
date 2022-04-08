import numpy as np


annot = np.load('./Fred/reannotations.npy')
for i in range(annot.shape[0]):
    print(annot[i])
    if i == 10:
        break
print(annot.shape)
