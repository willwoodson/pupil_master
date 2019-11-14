import os
import time

import cv2
import numpy as np
import pandas as pd

from library.eye_detect import EyeDetect
from library.world_detect import WorldDetect
from library.object_recognize import ObjectRecognize

data_video_record = "Data/data_video/show_video.avi"
data_csv = "Data/data_csv/object_distence.csv"

def restore_color(frame):
    b, g, r = cv2.split(frame)
    frame = cv2.merge([r, g, b])
    return frame


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


def voice(detect_world, recognize_object, board):

    board = 255 * np.ones([480, 640, 3], np.uint8)

    dict = {0.0: "mf: ",
            1.0: "cz: ",
            2.0: "sx: ",
            3.0: "zb: "}

    df = pd.DataFrame(
        np.zeros((4, 2)),
        columns=["object", "distence"],
    )

    for i in range(4):
        if recognize_object.df.iloc[i,0] != 500:
            df.iloc[i,0] = dict[recognize_object.df.iloc[i,0]]

            object_x = int(recognize_object.df.iloc[i, 1] + 0.5 * recognize_object.df.iloc[i, 3])
            object_y = int(recognize_object.df.iloc[i, 2] + 0.5 * recognize_object.df.iloc[i, 4])
            world_x = int(detect_world.world_x[0])
            world_y = int(detect_world.world_y[0])

            print("object_x:", object_x)
            print("object_y:", object_y)
            print("world_x:", world_x)
            print("world_y:", world_y)

            distence = (((object_x - world_x)**2) + ((object_y - world_y)**2))**0.5

            df.iloc[i,1] = round(distence, 6)

            print("distence:", df.iloc[i, 1])

        else:
            df.iloc[i, 0] = "None"
            df.iloc[i, 1] = 500.0

    df.to_csv(data_csv)
    df.sort_values(by = "distence", inplace=True)
    print(df)

    for i in range(4):
        if df.iloc[i,0] != "None":
            cv2.putText(
                board,
                df.iloc[i, 0],
                (50, 100 + 50*i),
                cv2.FONT_ITALIC,
                0.8,
                (100, 200, 80),
                2,
            )

            cv2.putText(
                board,
                str(df.iloc[i, 1]),
                (300, 100 + 50*i),
                cv2.FONT_ITALIC,
                0.8,
                (100, 200, 80),
                2,
            )

            if i == 0:
                cv2.putText(
                    board,
                    "you are looking " + df.iloc[i, 0],
                    (50, 50),
                    cv2.FONT_ITALIC,
                    0.8,
                    (100, 155, 80),
                    
                    2,
                )
        else:
            pass


    return board


webcam1 = cv2.VideoCapture(0) # world
webcam2 = cv2.VideoCapture(2) # eye


recognize_object = ObjectRecognize()

detect_eye = EyeDetect()
detect_world = WorldDetect()


# 定义编解码器并创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(data_video_record, fourcc, 20.0, (1280, 480))

df = pd.DataFrame(
    np.zeros((1, 6)),
    columns=["eye_x", "eye_y", "pipil_x", "pupil_y", "pupil_w", "pupil_h"],
)

board = 255*np.ones([480, 640, 3], np.uint8)


while True:
    _, frame1 = webcam1.read()
    _, frame2 = webcam2.read()


    world = cv2.resize(frame1, (640, 480))
    eye = cv2.resize(frame2, (640, 480))

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

    world, eye = draw_circle(world, eye, detect_world, detect_eye)

    # recognize_object.detect(object)
    # object = recognize_object.draw_circle(object)

    # board = voice(detect_world, recognize_object, board)

    output = np.hstack((world, eye))
    # output2 = np.hstack((object, board))
    # output = np.vstack((output1, output2))

    cv2.namedWindow("detect", cv2.WINDOW_AUTOSIZE)
    out.write(output)
    cv2.imshow("detect", output)

    if cv2.waitKey(1) == 27:
        break
    else:
        pass


k = cv2.waitKey(0)
cv2.destroyAllWindows()