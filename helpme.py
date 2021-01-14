import joblib
import matplotlib.pyplot as plt
with open("datas/study.jb", mode="rb") as f:
    x = joblib.load(f)

def sample_images(input_source_real):
        x,y=5,5#画像を5*5個
        #gen_imgs = 0.5 * gen_imgs + 0.5    # プロットのために0から1の範囲に収める
        # 5*5枚の画像を並べて1枚の画像にする
        fig, axs = plt.subplots(x, y)
        plt.subplots_adjust(wspace=0.4)
        cnt = 0
        for i in range(y):
            for j in range(x):
                axs[i,j].imshow(input_source_real[cnt], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        plt.show(block=False)
        plt.pause(20)
sample_images(x['mod'])