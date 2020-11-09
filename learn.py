from testnet import *
import joblib
import numpy as np
## 生成器（generator）を作る関数
noise_dim = 100 # ノイズの次元を指定
input_shape= (noise_dim,)
image_shape= (32,32,1) #生成する画像の大きさ
middle_size = 1000
#識別器を定義
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
#生成器を定義
generator = build_generator()

#ここからGANを定義（生成器のパラメタ更新に使う）
noise_for_image = Input(shape=(noise_dim,)) #GANに入力されるノイズ
fake_image = generator(noise_for_image) #生成器から作られる偽画像
discriminator.trainable = False #ここでは識別器は固定しておく（識別器は別のプロセスでパラメータ更新する）
outputs_GAN = discriminator(fake_image) #生成画像を識別器に入れて出てくる確率

#GANモデルを定義
GAN = keras.Model(inputs=noise_for_image, outputs=outputs_GAN, name="GAN")
GAN.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy']) 

GAN.summary()

#MNISTデータを読んでおく
xin=joblib.load("datas/sett1.jb")
xtest=joblib.load("datas/testt2.jb")
alpha=len(xin['mod'])
(x_train, y_train) =(np.array(xin['mod']),np.array(xin['origin']))
(x_test, y_test) =(np.array(xtest['mod']),np.array((xtest['origin'])))
x_train = x_train.reshape(alpha, 32, 32, 1).astype("float32") / 255 #画像をベクトルに変換
x_train = (x_train * 2) -1

#学習の設定
epochs = 100000
batch_size = 32
sample_interval = 3000 #画像を出力するepoch間隔
half_batch_size = int(batch_size / 2) #識別器の学習のために、バッチサイズの半分という値を準備

#学習開始
for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], half_batch_size) #データから画像をランダムに抽出
    real_images = x_train[idx]

    ##---ここから識別器の学習ステップ---
    noise = np.random.normal(0, 1, (half_batch_size, 100)) # 画像のためのノイズを生成
    generated_images = generator.predict(noise) #画像を生成

    # 本望の画像が半分、偽物の画像が半分のデータセットをつくる
    concat_data = np.concatenate([real_images, generated_images], axis=0) 
    concat_label = np.concatenate([np.ones((half_batch_size, 1)), np.zeros((half_batch_size, 1))], axis=0) #対応するラベル
    
    # 本物・偽物が混ざった画像データから識別器を訓練（本物・偽物のラベルを当てられるようにする） 
    d_loss, d_acc = discriminator.train_on_batch(concat_data, concat_label)
    
    ##---ここから生成器の学習ステップ---
    noise = np.random.normal(loc=0, scale=1, size=(batch_size, 100)) #画像のためのノイズを生成
    label = np.ones((batch_size, 1)) #全て1のラベルを作る（つまり全部が観測した本物の画像だとする）

    # GAN（実質的に生成器のパラメータ）を更新
    g_loss, g_acc = GAN.train_on_batch(noise, label) #ノイズを入れたときに、識別器が本物画像だと判定（ラベル1）を言うように学習
    
    # 一定間隔で生成した画像を表示
    if epoch % sample_interval == 0:
        print ("%d [d_loss: %f, d_acc:%s] [g_loss: %f, g_acc:%s]" % (epoch, d_loss, d_acc, g_loss, g_acc))
        sample_images(epoch)
print ("%d [d_loss: %f, d_acc:%s] [g_loss: %f, g_acc:%s]" % (epochs, d_loss, d_acc, g_loss, g_acc))
sample_images(epochs)