import cv2
import numpy
import dlib
import pandas as pd
import numpy as np
import time


class ObjectRecognize(object):

    def __init__(self):
        self.dict = {0.0: "mf", 1.0: "cz", 2.0: "sx",3.0: "zb"}
        detector1 = dlib.fhog_object_detector("model/classify/mf.svm")
        detector2 = dlib.fhog_object_detector("model/classify/cz.svm")
        detector3 = dlib.fhog_object_detector("model/classify/sx.svm")
        detector4 = dlib.fhog_object_detector("model/classify/zb.svm")
        self.detectors = [detector1, detector2, detector3, detector4]
        self.data_csv = "Data/data_csv/predict_object_recognize.csv"

        self.df = pd.DataFrame(
            500 * np.ones((4, 6)),
            columns=["name", "x", "y", "w", "h", "confidences"],
        )


    def detect(self, frame):
        t1 = time.time()

        self.df = pd.DataFrame(
            500 * np.ones((4, 6)),
            columns=["name", "x", "y", "w", "h", "confidences"],
        )

        [boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(self.detectors, frame,upsample_num_times=1,
                                                                                     adjust_threshold = 0.0)
        for i in range(len(boxes)):
            # print("detector {} found box {} with confidence {}.".format(detector_idxs[i], boxes[i], confidences[i]))
            self.df.iloc[i, 0] = detector_idxs[i]
            self.df.iloc[i, 1] = boxes[i].left()
            self.df.iloc[i, 2] = boxes[i].top()
            self.df.iloc[i, 3] = boxes[i].right() - boxes[i].left()
            self.df.iloc[i, 4] = boxes[i].bottom() - boxes[i].top()
            self.df.iloc[i, 5] = round(confidences[i], 6)

        self.df.to_csv(self.data_csv)
        t2 = time.time()
        print("检测耗时:", round(t2 - t1, 6))


    def draw_circle(self, frame):
        for i in range(4):
            if self.df.iloc[i, 0] != 500:
                cv2.putText(
                    frame, self.dict[self.df.iloc[i, 0]],
                    (int(self.df.iloc[i, 1]) + 10, int(self.df.iloc[i, 2]) - 10),
                    cv2.FONT_ITALIC,
                    0.6,
                    (10 + 20 * i, 20 + 30 * i, 200),
                    2,
                )

                cv2.rectangle(
                    frame,
                    (int(self.df.iloc[i, 1]), int(self.df.iloc[i, 2])),
                    (
                        int(self.df.iloc[i, 1]) + int(self.df.iloc[i, 3]),
                        int(self.df.iloc[i, 2]) + int(self.df.iloc[i, 4]),
                    ),
                    (10 + 20 * i, 20 + 30 * i, 200),
                    2,
                )
            else:
                pass

        return frame


