import os
import sys
import glob
import dlib

options = dlib.simple_object_detector_training_options()
# 单个眼睛不是左右对称的
# options.add_left_right_image_flips = True
# 支持向量机的C参数，通常默认取为5.自己适当更改参数以达到最好的效果
options.C = 5
# 线程数，你电脑有4核的话就填4
options.num_threads = 4
options.be_verbose = True


training_xml_path = "pupil.xml"
dlib.train_simple_object_detector(training_xml_path, "../model/pupil.svm", options)
print("")
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "../model/pupil.svm")))


training_xml_path = "eye.xml"
dlib.train_simple_object_detector(training_xml_path, "../model/eye.svm", options)
print("")
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "../model/eye.svm")))






