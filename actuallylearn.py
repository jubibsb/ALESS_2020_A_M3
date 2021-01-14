from new import *
import joblib
import numpy as np
from keras.layers import Concatenate
import warnings
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.transforms import Bbox
warnings.filterwarnings("ignore") #sorry m8
in_shape=(20,28)
image_shape=(28,28)
from keras.utils import plot_model


ITERATION_NUMBER=9
discriminator=build_discriminator()
discriminator.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(0.0002,0.5),metrics=['accuracy'])

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
        plt.savefig('datas/add_figure'+ITERATION_NUMBER+'_'+str(epoch)+'.png', bbox_inches='tight')
        #plt.show(block=False)
        #plt.pause(3)
        #plt.clf()
        #plt.close(fig)


for i in range(16,51):
	ITERATION_NUMBER=str(i+50)

	if (i)<=10:
		generator=build_generator_1(middle_dim1=int(100+20*(i)),middle_dim2=int(200+20*(i)),middle_dim3=int(100+20*(i)))#基本的に+200,10とか1は100だった
	elif (i-10)<=10:
		generator=build_generator_2(middle_dim1=int(100+20*(i-10)),middle_dim2=int(50+20*(i-10)),middle_dim3=int(10+20*(i-10)))
	elif (i-20)<=10:
		generator=build_generator_3(middle_dim1=int(200+20*(i-20)),middle_dim2=int(1+20*(i-20)),middle_dim3=int(10+20*(i-20)))
	elif (i-30)<=10:
		generator=build_generator_4(middle_dim1=int(100+20*(i-30)),middle_dim2=int(1+20*(i-30)),middle_dim3=int(10+20*(i-30)))
	elif (i-40)<=10:
		generator=build_generator_5(middle_dim1=int(100+20*(i-40)),middle_dim2=int(1+20*(i-40)),middle_dim3=int(10+20*(i-40)))

	with open('datas/add_report'+ITERATION_NUMBER+'.txt','w') as fh:
	   generator.summary(print_fn=lambda x: fh.write(x + '\n'))


	in_for_image= Input(shape=(20,28,))#x*y*num_data
	discriminator=build_discriminator()
	discriminator.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(0.0002,0.5),metrics=['accuracy'])
	fake_image_bottom=generator(in_for_image)
	discriminator.trainable=False
	fake_image=Concatenate(axis=1)([in_for_image,fake_image_bottom])
	outputs_GAN=discriminator(fake_image)

	GAN=Model(inputs=in_for_image,outputs=outputs_GAN,name="GAN")
	GAN.compile(loss="binary_crossentropy",optimizer=keras.optimizers.Adam(0.0002,0.5),metrics=['accuracy'])

	GAN.summary()
	plot_model(GAN, to_file='add_model_all_'+ITERATION_NUMBER+'.png',show_shapes=True)
	plot_model(generator, to_file='add_model_g_'+ITERATION_NUMBER+'.png',show_shapes=True)
	xin=joblib.load("datas/study.jb")
	xtest=joblib.load("datas/test.jb")
	length_in=len(xin["mod"])
	(x_train,y_train,label_train)=(np.array(xin['mod']),np.array(xin['origin']),to_categorical(np.array(xin['label']),20))
	(x_test,y_test,label_test)=(np.array(xtest['mod']),np.array(xtest['origin']),to_categorical(np.array(xtest['label']),20))



	epochs=50000
	batch_size=32
	sample_interval=10000
	half_batch_size=int(batch_size/2)
	#print(type(y_train))
	#学習
	with open("datas/add_results_"+ITERATION_NUMBER+".txt","w") as f:
		f.write("epoch,d_loss,d_acc,g_loss,g_acc\n")
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

		if epoch % (sample_interval/10) == 0:
			print ("%d [d_loss: %f, d_acc:%s] [g_loss: %f, g_acc:%s]" % (epoch, d_loss, d_acc, g_loss, g_acc))
			ids=np.array(np.random.randint(0, len(x_test),26))
			with open("datas/add_results_"+ITERATION_NUMBER+".txt","a") as f:
				f.write(str(epoch)+","+str(d_loss)+","+str(d_acc)+","+str(g_loss)+","+str(g_acc)+"\n")
		if epoch % (sample_interval) == 0:
			sample_images(epoch,x_test[ids],y_test[ids],label_test[ids])
			discriminator.save_weights('datas/add_param_D_'+ITERATION_NUMBER+"_"+str(epoch)+'.hdf5')
			GAN.save_weights('datas/add_param_G_'+ITERATION_NUMBER+"_"+str(epoch)+'.hdf5')


	print ("%d [d_loss: %f, d_acc:%s] [g_loss: %f, g_acc:%s]" % (epochs, d_loss, d_acc, g_loss, g_acc))
	sample_images(epoch,x_test[ids],y_test[ids],label_test[ids])
	with open("datas/add_results_"+ITERATION_NUMBER+".txt","a") as f:
		f.write(str(epoch+1)+","+str(d_loss)+","+str(d_acc)+","+str(g_loss)+","+str(g_acc)+"\n")

	discriminator.save_weights('datas/add_param_D_'+ITERATION_NUMBER+'.hdf5')
	GAN.save_weights('datas/add_param_G_'+ITERATION_NUMBER+'.hdf5')