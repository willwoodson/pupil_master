import os
import time

import cv2
import numpy as np
import pandas as pd

from library.eye_detect import EyeDetect
from library.world_detect import WorldDetect
from library.object_recognize import ObjectRecognize

data_video_record = "Data/data_video/main_d.avi"


def draw_circle(world, eye, detect_world, detect_eye):

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


webcam1 = cv2.VideoCapture(0)  # world
webcam2 = cv2.VideoCapture(1)  # eye


detect_eye = EyeDetect()
detect_world = WorldDetect()


# 定义编解码器并创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(data_video_record, fourcc, 20.0, (1280, 480))

df = pd.DataFrame(
    np.zeros((1, 6)),
    columns=["eye_x", "eye_y", "pipil_x", "pupil_y", "pupil_w", "pupil_h"],
)


while True:
    _, frame1 = webcam1.read()
    _, frame2 = webcam2.read()

    world = cv2.resize(frame1, (640, 480))
    eye = cv2.resize(frame2, (640, 480))

    detect_eye.detect(eye)
    (
        df.iloc[0, 0],
        df.iloc[0, 1],
        df.iloc[0, 2],
        df.iloc[0, 3],
        df.iloc[0, 4],
        df.iloc[0, 5],
    ) = (
        detect_eye.eye_x,
        detect_eye.eye_y,
        detect_eye.pupil_x,
        detect_eye.pupil_y,
        detect_eye.pupil_w,
        detect_eye.pupil_h,
    )

    detect_world.detect(df)

    world, eye = draw_circle(world, eye, detect_world, detect_eye)

    output = np.hstack((world, eye))

    cv2.namedWindow("detect", cv2.WINDOW_AUTOSIZE)
    out.write(output)
    cv2.imshow("detect", output)

    if cv2.waitKey(1) == 27:
        break
    else:
        pass


k = cv2.waitKey(0)
cv2.destroyAllWindows()
