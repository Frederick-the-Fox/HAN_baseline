import scipy.io as sio
from sklearn.preprocessing import OneHotEncoder

# def encode_onehot(labels):
#     labels = labels.reshape(-1, 1)
#     enc = OneHotEncoder()
#     enc.fit(labels)
#     labels_onehot = enc.transform(labels).toarray()
#     return labels_onehot

# path = "/home/hangni/WangYC/HAN/data/imdb/imdb5k.mat"
path = '//home/hangni/WangYC/HAN/data/acm/ACM3025.mat'
data = sio.loadmat(path)
# print(type(data))
print(data.keys())

feature = data['feature']
print(feature.shape)
# print(feature)
# print(feature.shape)

# labels = data['label']
# print(labels)
# print(labels.shape)

# labels_onehot = encode_onehot(labels)
# print(labels_onehot)
# print(labels_onehot.shape)
