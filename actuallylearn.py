from new import *
import joblib
import numpy as np
from keras.layers import Concatenate
import warnings
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.transforms import Bbox
warnings.filterwarnings("ignore")
in_shape=(14,28)
image_shape=(28,28)


discriminator=build_discriminator()
discriminator.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(0.0002,0.5),metrics=['accuracy'])

generator=build_generator()

def sample_images(epoch,input_source_in,input_source_real,label_real):

    x,y=5,5#画像を5*5個
    gen_imgs = np.concatenate([input_source_in,generator.predict(input_source_in)],axis=1) # generatorにノイズを入れて画像生成
    #gen_imgs = 0.5 * gen_imgs + 0.5    # プロットのために0から1の範囲に収める
    # 5*5枚の画像を並べて1枚の画像にする
    label_predict=discriminator.predict(gen_imgs)
    fig, axs = plt.subplots(x, y)
    plt.subplots_adjust(wspace=0.4)
    cnt = 0
    for i in range(y):
        for j in range(x):
            axs[i,j].imshow(np.concatenate([input_source_real[cnt],gen_imgs[cnt, :,:]],axis=1), cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
            #axs[i,j].title.set_text(str(np.argmax(label_real[cnt]))+","+str(np.argmax(label_predict[cnt])))
            #at = AnchoredSizeBar(axs[i,j].transData,size=20
            #    label=str(np.argmax(label_real[cnt]))+","+str(np.argmax(label_predict[cnt])),
            #    frameon=False,bbox_to_anchor=Bbox.from_bounds(0, 0, 1, 1),
            #    loc='upper left',bbox_transform=axs[i,j].figure.transFigure
            #    )
            #at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            #axs[i,j].add_artist(at)
    plt.savefig('datas/figure_'+str(epoch)+'.png', bbox_inches='tight')
    plt.show(block=False)
    plt.pause(3)
    plt.clf()
    plt.close(fig)

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
(x_train,y_train,label_train)=(np.array(xin['mod']),np.array(xin['origin']),to_categorical(np.array(xin['label']),20))
(x_test,y_test,label_test)=(np.array(xtest['mod']),np.array(xtest['origin']),to_categorical(np.array(xtest['label']),20))


epochs=500000
batch_size=32
sample_interval=10000
half_batch_size=int(batch_size/2)
#print(type(y_train))
#学習
    
for epoch in range(epochs):
    idx = np.array(np.random.randint(0, len(x_train), half_batch_size)) #データから画像をランダムに抽出
    real_images = y_train[idx]
    intrain=x_train[idx]
    in_labeltrain=label_train[idx]
    #print(x_train.shape)
    #print(generator.predict(intrain).shape)
    generated_images=np.concatenate([x_train[idx],generator.predict(intrain)],axis=1)
    #print(generated_images.shape)
    #print(real_images.shape)
    concat_data=np.concatenate([real_images, generated_images], axis=0)
    concat_label=np.concatenate([in_labeltrain,to_categorical(np.array(xin['label'])[idx]+10,20)])

    d_loss,d_acc=discriminator.train_on_batch(concat_data,concat_label)

    gidx = np.random.randint(0, len(x_test), half_batch_size) #データから画像をランダムに抽出
    g_real_images = y_train[gidx]
    g_in=x_train[gidx]
    g_label=label_test[gidx]

    g_loss,g_acc=GAN.train_on_batch(g_in,g_label)

    if epoch % sample_interval == 0:
        print ("%d [d_loss: %f, d_acc:%s] [g_loss: %f, g_acc:%s]" % (epoch, d_loss, d_acc, g_loss, g_acc))
        ids=np.array(np.random.randint(0, len(x_test),26))
        sample_images(epoch,x_test[ids],y_test[ids],label_test[ids])
        discriminator.save_weights('datas/param_D_1_'+str(epoch)+'.hdf5')
        GAN.save_weights('datas/param_G_1_'+str(epoch)+'.hdf5')
        with open("datas/results_1_"+str(epoch)+".txt","a") as f:
            f.write("epoch,d_loss,d_acc,g_loss,g_acc\n")
            f.write(str(epoch)+","+str(d_loss)+","+str(d_acc)+","+str(g_loss)+","+str(g_acc))


print ("%d [d_loss: %f, d_acc:%s] [g_loss: %f, g_acc:%s]" % (epochs, d_loss, d_acc, g_loss, g_acc))
sample_images(epoch,x_test[ids],y_test[ids],label_test[ids])

discriminator.save_weights('param_D.hdf5')
GAN.save_weights('param_G.hdf5')