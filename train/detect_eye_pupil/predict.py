import os
import cv2
import glob
import dlib

detector_pupil = dlib.simple_object_detector("../model/pupil.svm")
detector_eye = dlib.simple_object_detector("../model/eye.svm")

test_folder = "../data_img/eyes/"

# 定义编解码器并创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("../data_video/predict_eye_pupil.avi", fourcc, 20.0, (640, 480))

for f in glob.glob(test_folder + "*.jpg"):
    print("Processing file: {}".format(f))
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    dets_pupil = detector_pupil(img2)
    dets_eye = detector_eye(img2)
    print("Number of pupils detected: {}".format(len(dets_pupil)))
    for index, pupil in enumerate(dets_pupil):
        print(dets_pupil)
        print(
            "pupil {}; left {}; top {}; right {}; bottom {}".format(
                index, pupil.left(), pupil.top(), pupil.right(), pupil.bottom()
            )
        )

        left = pupil.left()
        top = pupil.top()
        right = pupil.right()
        bottom = pupil.bottom()

        pupil_x = int((right + left) / 2)
        pupil_y = int((top + bottom) / 2)

        cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 0), 3)

        # 画十字标
        color = (0, 255, 0)

        cv2.line(
            img, (pupil_x - 30, pupil_y), (pupil_x + 30, pupil_y), color, thickness=2
        )
        cv2.line(
            img, (pupil_x, pupil_y - 30), (pupil_x, pupil_y + 30), color, thickness=2
        )

    print("Number of eyes detected: {}".format(len(dets_eye)))
    for index, eye in enumerate(dets_eye):
        print(dets_eye)
        print(
            "eye {}; left {}; top {}; right {}; bottom {}".format(
                index, eye.left(), eye.top(), eye.right(), eye.bottom()
            )
        )

        left = eye.left()
        top = eye.top()
        right = eye.right()
        bottom = eye.bottom()

        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)

    cv2.namedWindow("detect", cv2.WINDOW_AUTOSIZE)

    # 保存当前帧
    out.write(img)
    cv2.imwrite("../data_video/predict_eye_pupil.jpg", img)

    cv2.imshow("detect", img)
    if cv2.waitKey(1) == 27:
        break
    else:
        pass


k = cv2.waitKey(0)
cv2.destroyAllWindows()
