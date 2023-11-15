import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import pickle as pkl

data = scipy.io.loadmat("newnoy/p2dnmr.mat")['data']
imag_data3D = np.imag(data)
# print(imag_data3D.sum())
# plt.imshow(imag_data3D)
# plt.figure()
with open('nmr11.pkl', "rb") as f:
    re = pkl.load(f)
re_full = re.full().numpy()
plt.imshow(np.abs(re_full-imag_data3D)/np.max(np.abs(imag_data3D)))
plt.colorbar()
plt.show()