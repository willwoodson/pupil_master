import cv2
import numpy
import dlib
import pandas as pd
import numpy as np
import time

df = pd.DataFrame(
    100*np.ones((4, 6)),
    columns=["name", "x", "y", "w", "h", "confidences"],
)

dict = {0.0: "mf", 1.0: "cz", 2.0: "sx",3.0: "zb"}

# 定义编解码器并创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("../data_video/predict_object_recognize.avi", fourcc, 20.0, (1280, 480))

detector1 = dlib.fhog_object_detector("model/mf.svm")
detector2 = dlib.fhog_object_detector("model/cz.svm")
detector3 = dlib.fhog_object_detector("model/sx.svm")
detector4 = dlib.fhog_object_detector("model/zb.svm")

detectors = [detector1, detector2,detector3, detector4]


def draw_circle(frame, df):
    global dict
    for i in range(4):
        if df.iloc[i,0] != 100:
            cv2.putText(
                frame, dict[df.iloc[i,0]],
                (int(df.iloc[i,1])+ 10, int(df.iloc[i,2])-10),
                cv2.FONT_ITALIC,
                0.6,
                (10+20*i, 20+30*i, 200),
                2,
            )

            cv2.rectangle(
                frame,
                (int(df.iloc[i,1]), int(df.iloc[i,2])),
                (
                    int(df.iloc[i,1]) + int(df.iloc[i,3]),
                    int(df.iloc[i,2]) + int(df.iloc[i,4]),
                ),
                (10+20*i, 20+30*i, 200),
                2,
            )
        else:
            pass

    return frame

cap = cv2.VideoCapture("../data_video/test.mp4")  # 参数为视频文件目录

while True:
    ret, frame = cap.read()
    image = frame[:, 0:640]

    t1 = time.time()

    df = pd.DataFrame(
        100 * np.ones((4, 6)),
        columns=["name", "x", "y", "w", "h", "confidences"],
    )

    [boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectors, image, upsample_num_times=1,
                                                                                 adjust_threshold=0.0)
    for i in range(len(boxes)):
        # print("detector {} found box {} with confidence {}.".format(detector_idxs[i], boxes[i], confidences[i]))
        df.iloc[i, 0] = detector_idxs[i]
        df.iloc[i, 1] = boxes[i].left()
        df.iloc[i, 2] = boxes[i].top()
        df.iloc[i, 3] = boxes[i].right() - boxes[i].left()
        df.iloc[i, 4] = boxes[i].bottom() - boxes[i].top()
        df.iloc[i, 5] = round(confidences[i], 6)

    df.to_csv("../data_csv/predict_object_recognize.csv")
    t2 = time.time()
    print("检测耗时:", t2 - t1)

    frame = draw_circle(frame.copy(), df)

    cv2.namedWindow("detect", cv2.WINDOW_AUTOSIZE)

    # 保存当前帧
    out.write(frame)

    cv2.imshow("detect", frame)

    if cv2.waitKey(10) == 27:
        break
    else:
        pass


cap.release()
cv2.destroyAllWindows()