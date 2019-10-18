import os
import time

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from area_recognize import AreaRecognize
from eye_detect import EyeDetect
from world_detect import WorldDetect

test_folder = "data_img/origin"
detect_eye = EyeDetect()
detect_world = WorldDetect()
recognize_area = AreaRecognize()

# 定义编解码器并创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("data_video/detect_origin_c.avi", fourcc, 20.0, (1760, 480))

df = pd.DataFrame(
    np.zeros((1, 6)),
    columns=["eye_x", "eye_y", "pipil_x", "pupil_y", "pupil_w", "pupil_h"],
)

file_list = os.listdir(test_folder)
num = len(file_list)
count = 0


def restore_color(frame):
    b, g, r = cv2.split(frame)
    frame = cv2.merge([r, g, b])
    return frame


def draw_circle(world, eye, detect_world, detect_eye, recognize_area):

    cv2.circle(world, (detect_world.world_x, detect_world.world_y), 30, (0, 255, 0), 5)
    # 画十字标
    cv2.line(
        world,
        (detect_world.world_x - 10, detect_world.world_y),
        (detect_world.world_x + 10, detect_world.world_y),
        (255, 255, 0),
        thickness=5,
    )
    cv2.line(
        world,
        (detect_world.world_x, detect_world.world_y - 10),
        (detect_world.world_x, detect_world.world_y + 10),
        (255, 255, 0),
        thickness=5,
    )

    cv2.putText(
        world,
        recognize_area.rec_result.iloc[-1, 0],
        (detect_world.world_x - 50, detect_world.world_y - 50),
        cv2.FONT_ITALIC,
        0.8,
        (100, 200, 80),
        2,
    )

    cv2.rectangle(
        eye,
        (detect_eye.pupil_x, detect_eye.pupil_y),
        (
            detect_eye.pupil_x + detect_eye.pupil_w,
            detect_eye.pupil_y + detect_eye.pupil_h,
        ),
        (150, 255, 0),
        3,
    )
    cv2.rectangle(
        eye,
        (detect_eye.eye_x, detect_eye.eye_y),
        (detect_eye.eye_x + detect_eye.eye_w, detect_eye.eye_y + detect_eye.eye_h),
        (0, 255, 0),
        3,
    )
    # 画十字标
    cv2.line(
        eye,
        (detect_eye.pupil_c_x - 30, detect_eye.pupil_c_y),
        (detect_eye.pupil_c_x + 30, detect_eye.pupil_c_y),
        (255, 255, 0),
        thickness=2,
    )
    cv2.line(
        eye,
        (detect_eye.pupil_c_x, detect_eye.pupil_c_y - 30),
        (detect_eye.pupil_c_x, detect_eye.pupil_c_y + 30),
        (255, 255, 0),
        thickness=2,
    )

    return world, eye


def get_roi(world, detect_world):
    left = int(detect_world.world_x - 40) if detect_world.world_x - 40 > 0 else 0
    right = int(detect_world.world_x + 40)
    top = int(detect_world.world_y - 40) if detect_world.world_y - 40 > 0 else 0
    bottom = int(detect_world.world_y + 40)
    roi = world[top:bottom, left:right]
    cv2.imwrite("data_img/cam/mid.jpg", roi)
    roi = cv2.resize(roi, (480, 480))
    return roi


with tf.Session() as sess:
    for files in file_list:
        t1 = time.time()
        img_dir = os.path.join(test_folder, files)
        print("Processing file: {}".format(img_dir))
        print("这是第" + str(count + 1) + "张图片, 共有" + str(num) + "张图片")

        frame = cv2.imread(img_dir, cv2.IMREAD_COLOR)
        # frame = restore_color(frame).copy()

        world = frame[:, 0:640]
        eye = frame[:, 640:1280]
        roi = world.copy()

        detect_eye.detect(eye)
        df.iloc[0, 0], df.iloc[0, 1], df.iloc[0, 2], df.iloc[0, 3], df.iloc[
            0, 4
        ], df.iloc[0, 5] = (
            detect_eye.eye_x,
            detect_eye.eye_y,
            detect_eye.pupil_x,
            detect_eye.pupil_y,
            detect_eye.pupil_w,
            detect_eye.pupil_h,
        )

        detect_world.detect(df)

        roi = get_roi(roi, detect_world)
        t1 = time.time()
        recognize_area.recognize("data_img/cam/mid.jpg", sess)
        t2 = time.time()
        print("计算耗时：", t2 - t1)

        world, eye = draw_circle(world, eye, detect_world, detect_eye, recognize_area)

        output = np.hstack((world, eye))
        output = np.hstack((output, roi))

        cv2.namedWindow("detect", cv2.WINDOW_AUTOSIZE)
        out.write(output)
        cv2.imshow("detect", output)

        if cv2.waitKey(1) == 27:
            break
        else:
            pass

        count += 1


k = cv2.waitKey(0)
cv2.destroyAllWindows()
