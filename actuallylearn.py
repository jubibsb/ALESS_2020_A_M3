from new import *
import joblib
import numpy as np
from keras.layers import Concatenate

in_shape=(14,28)
image_shape=(28,28)

discriminator=build_discriminator()
discriminator.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(0.0002,0.5),metrics=['accuracy'])

generator=build_generator()


in_for_image= Input(shape=(14,28,))#x*y*num_data
fake_image_bottom=generator(in_for_image)
discriminator.trainable=False
fake_image=Concatenate(axis=1)([in_for_image,fake_image_bottom])
outputs_GAN=discriminator(fake_image)

GAN=Model(inputs=in_for_image,outputs=outputs_GAN,name="GAN")
GAN.compile(loss="binary_crossentropy",optimizer=keras.optimizers.Adam(0.0002,0.5),metrics=['accuracy'])

GAN.summary()

xin=joblib.load("datas/study.jb")
xtest=joblib.load("datas/test.jb")
length_in=len(xin["mod"])
(x_train,y_train,label_train)=(xin['mod'],xin['origin'],to_categorical(xin['label'],11))
(x_test,y_test,label_test)=(xtest['mod'],xtest['origin'],to_categorical(xtest['label'],11))


epochs=1000000
batch_size=32
sample_interval=50000
half_batch_size=int(batch_size/2)

#学習
for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], half_batch_size) #データから画像をランダムに抽出
    real_images = y_train[idx]
    intrain=x_train[idx]
    in_labeltrain=label_train[idx]
    generated_images=np.concatenate([x_train,generator.predict(intrain)],axis=0)

    concat_data=np.concatenate([real_images, generated_images], axis=0)
    concat_label=np.concatenate([in_labeltrain,to_categorical(np.full((half_batch_size,1),10))])

    d_loss,d_acc=discriminator.train_on_batch(concat_data,concat_label)

    gidx = np.random.randint(0, x_train.shape[0], half_batch_size) #データから画像をランダムに抽出
    g_real_images = y_train[gidx]
    g_in=x_train[gidx]
    g_label=label_train[gidx]

    g_loss,g_acc=GAN.train_on_batch(g_in,g_label)

    if epoch % sample_interval == 0:
        print ("%d [d_loss: %f, d_acc:%s] [g_loss: %f, g_acc:%s]" % (epoch, d_loss, d_acc, g_loss, g_acc))
        sample_images(epoch)

print ("%d [d_loss: %f, d_acc:%s] [g_loss: %f, g_acc:%s]" % (epochs, d_loss, d_acc, g_loss, g_acc))
sample_images(epochs)

discriminator.save_weights('param_D.hdf5')
GAN.save_weights('param_G.hdf5')