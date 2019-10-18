import cv2
import time


webcam = cv2.VideoCapture(1)

while True:
    # 我们从网络摄像头中得到一个新的画面
    _, frame = webcam.read()


    if cv2.waitKey(10) == ord('s'):
        name = "../data_img/cam" + str(time.time())+ ".jpg"
        cv2.imwrite(name, frame)
    elif cv2.waitKey(10) == 27:
        break
    else:
        pass

    cv2.imshow("Demo", frame)



cv2.des troyAllWindows()