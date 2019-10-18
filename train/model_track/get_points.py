import os
import cv2
import glob
import pandas as pd
import numpy as np
from eye_detect import EyeDetect
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image
import time


test_folder = "../data_img/origin"
detect = EyeDetect()

file_list = os.listdir(test_folder)
num = len(file_list)
df = pd.DataFrame(np.zeros((num, 10)), columns=[
    'world_x', 'world_y', 'eye_x', 'eye_y', 'eye_w', 'eye_h', 'pipil_x', 'pupil_y', 'pupil_w', 'pupil_h'])
count = 0
fresh = 0

def on_press(event):
    global df, fresh, count
    print(event.button)
    if event.button == 1:
        fresh = 1
        df.iloc[count, 0] = int(event.xdata)
        df.iloc[count, 1] = int(event.ydata)
        print("左：", df.iloc[count, 0], df.iloc[count, 1])
    else:
        print("请点击关键点")

    if fresh == 1:
        fresh = 0
        time.sleep(0.5)
        plt.close(1)
    else:
        pass


for files in file_list:
    img_dir = os.path.join(test_folder, files)
    print("Processing file: {}".format(img_dir))
    print("这是第" + str(count+1) + "张图片, 共有" + str(num) + "张图片")
    frame = cv2.imread(img_dir, cv2.IMREAD_COLOR)

    img = frame[:, 0:640]
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(img, animated=True)
    fig.canvas.mpl_connect('button_press_event', on_press)
    plt.show()

    frame = frame[:, 640:1280]
    detect.detect(frame)

    detect.show(frame)
    cv2.namedWindow("detect", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("detect", frame)
    cv2.waitKey(1)

    df.iloc[count, 2], df.iloc[count, 3], df.iloc[count, 4], df.iloc[count, 5], df.iloc[count, 6], df.iloc[count, 7],df.iloc[count, 8], df.iloc[count, 9] = \
        detect.eye_x, detect.eye_y, detect.eye_w, detect.eye_h, detect.pupil_x, detect.pupil_y, detect.pupil_w, detect.pupil_h
    df.to_csv("../csv_data/points.csv")
    count+=1


k = cv2.waitKey(0)
cv2.destroyAllWindows()