import cv2
import time
import numpy as np

webcam1 = cv2.VideoCapture(2) # world
webcam2 = cv2.VideoCapture(1) # eye

# 定义编解码器并创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("../data_video/test.avi", fourcc, 20.0, (1280, 480))

while True:
    # 我们从网络摄像头中得到一个新的画面
    _, frame1 = webcam1.read()
    _, frame2 = webcam2.read()

    world = cv2.resize(frame1, (640, 480))
    eye = cv2.resize(frame2, (640, 480))

    frame = np.hstack((world, eye))
    out.write(frame)

    cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("test", frame)

    if cv2.waitKey(1) == 27:
        break
    else:
        pass



cv2.destroyAllWindows()