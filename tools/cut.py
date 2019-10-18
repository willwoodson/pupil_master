import os
import cv2

def cut():
    path = input("请输入路径：")
    output_path1 = input("请输入输出路径1：")
    output_path2 = input("请输入输出路径2：")
    count = 0
    file_list = os.listdir(path)
    for files in file_list:
        img_dir = os.path.join(path, files)
        img1_dir = os.path.join(output_path1, files)
        img2_dir = os.path.join(output_path2, files)
        img = cv2.imread(img_dir)
        img1 = img[:, 0:640]
        img2 = img[:, 640:1280]
        cv2.imwrite(img1_dir, img1)
        cv2.imwrite(img2_dir, img2)
        count += 1
    print("一共修改了"+str(count)+"个文件")


cut()