import os
import cv2
import glob
import dlib
import time
import pandas as pd
import numpy as np


test_folder = "img/"

df = pd.DataFrame(
    100*np.ones((6, 6)),
    columns=["name", "x", "y", "w", "h", "confidences"],
)

dict = {0.0: "sz", 1.0: "cz", 2.0: "dp",3.0: "pd", 4.0: "zb", 5.0: "mf"}

# 定义编解码器并创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("../data_video/predict_object_recognize.avi", fourcc, 20.0, (640, 480))

detector1 = dlib.fhog_object_detector("../model/sz.svm")
detector2 = dlib.fhog_object_detector("../model/cz.svm")
detector3 = dlib.fhog_object_detector("../model/dp.svm")
detector4 = dlib.fhog_object_detector("../model/pb.svm")
detector5 = dlib.fhog_object_detector("../model/zb.svm")
detector6 = dlib.fhog_object_detector("../model/mf.svm")

detectors = [detector1, detector2,detector3, detector4,detector5, detector6]


def draw_circle(frame, df):
    global dict
    for i in range(6):
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


for f in glob.glob(test_folder + "*.jpg"):
    print("Processing file: {}".format(f))
    image = cv2.imread(f, cv2.IMREAD_COLOR)
    # b, g, r = cv2.split(img)
    # image = cv2.merge([r, g, b])

    t1 = time.time()

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

    frame = draw_circle(image.copy(), df)

    cv2.namedWindow("detect", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("origin", cv2.WINDOW_AUTOSIZE)

    # 保存当前帧
    out.write(frame)

    cv2.imshow("origin", image)
    cv2.imshow("detect", frame)

    if cv2.waitKey(10) == 27:
        break
    else:
        pass


k = cv2.waitKey(0)
cv2.destroyAllWindows()
