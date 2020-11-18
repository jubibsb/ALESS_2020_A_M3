from keras.layers import Input 
from keras.layers import Dense 
from keras.models import Model 
from keras.layers import Conv2D, MaxPooling2D, Flatten 
from keras.utils import to_categorical
from keras.layers import BatchNormalization, Reshape, LeakyReLU, Flatten
import numpy as np
import keras
import matplotlib.pyplot as plt
## 生成器（generator）を作る関数
noise_dim = 100 # ノイズの次元を指定
input_shape= (noise_dim,)
image_shape= (32,32,1) #生成する画像の大きさ
middle_size = 1000

def build_generator():
  inputs = Input(shape=input_shape, name="Input") #入力はノイズ（ベクトル）
  #----------ここからモデル----------
  middle = Dense(middle_size)(inputs)
  middle = LeakyReLU(alpha=0.2)(middle) #LeakyReLU活性化関数はDenseで指定できないので、個別にここで指定
  middle = BatchNormalization(momentum=0.8)(middle) #バッチ正規化を入れる

  middle = Dense(middle_size)(middle)
  middle = LeakyReLU(alpha=0.2)(middle)
  middle = BatchNormalization(momentum=0.8)(middle)

  middle = Dense(middle_size)(middle)
  middle = LeakyReLU(alpha=0.2)(middle)
  middle = BatchNormalization(momentum=0.8)(middle)

  middle = Dense(np.prod(image_shape), activation="tanh")(middle) #生成する画像と同じ要素数を持つベクトルを準備
  outputs = Reshape(image_shape)(middle) #ベクトルを画像に並び替え
  #----------ここまでモデル----------
  generator = keras.Model(inputs=inputs, outputs=outputs, name="Generator")
  return generator #定義したモデルを返す
generator = build_generator()
generator.summary()

## 識別器（discriminator）を作る関数

def build_discriminator():
  inputs = Input(shape=image_shape) #入力は画像
  #----------ここからモデル----------
  middle = Flatten()(inputs) #入力された画像をベクトルに並び替え

  middle = Dense(middle_size)(middle)
  middle = LeakyReLU(alpha=0.2)(middle)
  
  middle = Dense(middle_size)(middle)
  middle = LeakyReLU(alpha=0.2)(middle)

  outputs = Dense(1, activation="sigmoid")(middle) #入力された画像が本物である確率（スカラー）を出力（sigmoid活性化関数を通しているので確率とみなせる値が出る）
  #----------ここまでモデル----------

  discriminator = keras.Model(inputs=inputs, outputs=outputs, name="Discriminator")
  return discriminator
discriminator = build_discriminator()
discriminator.summary()

#準備：途中で生成した画像を出力する関数
def sample_images(epoch):
  r, c = 5, 5 #生成する画像の数（5*5=25枚生成する）
  noise = np.random.normal(loc=0, scale=1, size=(r * c, 100))   #画像の生成のためのノイズを作る
  gen_imgs = generator.predict(noise) # generatorにノイズを入れて画像生成
  gen_imgs = 0.5 * gen_imgs + 0.5    # プロットのために0から1の範囲に収める
  # 5*5枚の画像を並べて1枚の画像にする
  fig, axs = plt.subplots(r, c)
  cnt = 0
  for i in range(r):
    for j in range(c):
      axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
      axs[i,j].axis('off')
      cnt += 1            
  plt.show()

  ## GANモデルを定義する

