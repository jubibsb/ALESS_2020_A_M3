import joblib
import matplotlib.pyplot as plt
with open("datas/set11.jb", mode="rb") as f:
    x = joblib.load(f)

plt.imshow(x['mod'][1],cmap = "gray")
plt.show()