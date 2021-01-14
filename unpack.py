import pickle
import joblib
import numpy as np
from mnist import MNIST

mndata=MNIST('samples')
s_images,s_labels=mndata.load_training()
t_images,t_labels=mndata.load_testing()
#ともにlist
num_image_s=len(s_images)
num_image_t=len(t_images)
s_images=np.array(s_images)
s_labels=np.array(s_labels)
t_images=np.array(t_images)
t_labels=np.array(t_labels)

data_origin_s=[]
data_mod_s=[]
data_label_s=[]

data_origin_t=[]
data_mod_t=[]
data_label_t=[]

for j in range(num_image_s):
	data_label_s.append(s_labels[j])
	data_origin_s.append(s_images[j].reshape((28,28)))
	data_mod_s.append(s_images[j].reshape((28,28))[0:20])#もともと0:14

for j in range(num_image_t):
	data_label_t.append(t_labels[j])
	data_origin_t.append(t_images[j].reshape((28,28)))
	data_mod_t.append(t_images[j].reshape((28,28))[0:20])#もともと0:14

dataset_s=dict(zip(["origin","mod","label"],[data_origin_s,data_mod_s,data_label_s]))
dataset_t=dict(zip(["origin","mod","label"],[data_origin_t,data_mod_t,data_label_t]))
joblib.dump(dataset_s, "datas/study.jb", compress=3)
joblib.dump(dataset_t, "datas/test.jb", compress=3)
