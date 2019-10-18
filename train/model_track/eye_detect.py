import dlib
import cv2


class EyeDetect(object):
    def __init__(self,
                 model_pupil="../model/pupil.svm",
                 model_eye="../model/eye.svm"):
        self.detector_pupil = dlib.simple_object_detector(model_pupil)
        self.detector_eye = dlib.simple_object_detector(model_eye)


    def detect(self,frame):
        b, g, r = cv2.split(frame)
        frame = cv2.merge([r, g, b])
        dets_pupil = self.detector_pupil(frame)
        dets_eye = self.detector_eye(frame)

        # print("Number of pupils detected: {}".format(len(dets_pupil)))
        for index, pupil in enumerate(dets_pupil):
            # print(dets_pupil)
            # print('pupil {}; left {}; top {}; right {}; bottom {}'.format(
            #     index, pupil.left(), pupil.top(), pupil.right(),pupil.bottom()))

            self.pupil_c_x = int((pupil.right() + pupil.left()) / 2)
            self.pupil_c_y = int((pupil.top() + pupil.bottom()) / 2)

            self.pupil_x = pupil.left()
            self.pupil_y = pupil.top()
            self.pupil_w = pupil.right() - pupil.left()
            self.pupil_h = pupil.bottom() - pupil.top()

        # print("Number of eyes detected: {}".format(len(dets_eye)))
        for index, eye in enumerate(dets_eye):
            # print(dets_eye)
            # print('eye {}; left {}; top {}; right {}; bottom {}'.format(
            #     index, eye.left(), eye.top(), eye.right(), eye.bottom()))

            self.eye_x = eye.left()
            self.eye_y = eye.top()
            self.eye_w = eye.right() - eye.left()
            self.eye_h = eye.bottom() - eye.top()


    def show(self,frame):
        cv2.rectangle(frame, (self.pupil_x, self.pupil_y),
                      (self.pupil_x+self.pupil_w, self.pupil_y+self.pupil_h), (150, 255, 0), 3)

        # 画十字标
        color = (255, 255, 0)

        cv2.line(frame, (self.pupil_c_x - 30, self.pupil_c_y), (self.pupil_c_x + 30, self.pupil_c_y), color, thickness=2)
        cv2.line(frame, (self.pupil_c_x, self.pupil_c_y - 30), (self.pupil_c_x, self.pupil_c_y + 30), color, thickness=2)

        cv2.rectangle(frame, (self.eye_x, self.eye_y),
                      (self.eye_x+self.eye_w, self.eye_y+self.eye_h), (0, 255, 0), 3)

        # cv2.namedWindow("detect", cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("detect", frame)
        # cv2.waitKey(0)
