import numpy as np
import pandas as pd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image
import os
import time


count_l, count_r = 0, 0
count, num = 0, 0
df = pd.DataFrame(np.zeros((20,4)),columns=['world_x','world_y','eye_x','eye_y'])

def on_press(event):
    global count_l, count_r, count, df, num
    # print(df)
    print(event.button)
    if event.button == 1:
        count_l = 1
        df.iloc[count - 1, 0] = int(event.xdata)
        df.iloc[count - 1, 1] = int(event.ydata)
        print("左：", df.iloc[count - 1, 0], df.iloc[count - 1, 1])
    elif event.button == 3:
        count_r = 1
        df.iloc[count - 1, 2] = int(event.xdata)-640
        df.iloc[count - 1, 3] = int(event.ydata)
        print("右：", df.iloc[count - 1, 2], df.iloc[count - 1, 3])
    else:
        print("请点击关键点")

    if count_l == 1 and count_r == 1:
        count_l, count_r = 0, 0
        time.sleep(0.5)
        # df.to_csv("../csv_data/points.csv")
        plt.close(1)
    else:
        pass


def get_point():
    global count, num, df
    path = input("请输入图片所在路径：")
    # path = "../test"
    file_list = os.listdir(path)
    num = len(file_list)
    df = pd.DataFrame(np.zeros((num, 4)), columns=['world_x', 'world_y', 'eye_x', 'eye_y'])
    for files in file_list:
        count += 1
        print(files)
        print("这是第"+str(count)+"张图片, 共有"+str(num)+"张图片")

        img_dir = os.path.join(path, files)

        fig = plt.figure(figsize=(12, 6))
        img = Image.open(img_dir)

        plt.imshow(img, animated=True)
        fig.canvas.mpl_connect('button_press_event', on_press)
        plt.show()


get_point()
print(df)
# df.to_csv("../csv_data/points.csv")

