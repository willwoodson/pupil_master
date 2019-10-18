from sklearn.externals import joblib
import pandas as pd


# 读取数据
point = pd.read_csv("../data_csv/points.csv")
eye = point[['eye_x', 'eye_y', 'pipil_x', 'pupil_y', 'pupil_w', 'pupil_h']]

# 导入模型
clf_world_x = joblib.load("../model/world_x.pkl")
clf_world_y = joblib.load("../model/world_y.pkl")
world_x = []
world_y = []

# 得到待预测的目标值
list = clf_world_x.predict(eye)
point["world_x_p"] = list
list = clf_world_y.predict(eye)
point["world_y_p"] = list
point.to_csv("../data_csv/predict_track.csv", encoding="utf-8")
print("success")


