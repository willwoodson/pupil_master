from sklearn.linear_model import SGDRegressor as SGDR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor as GBR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib
import pandas as pd


# 读取数据
point = pd.read_csv("../data_csv/points.csv")
world_x = point["world_x"]
world_y = point["world_y"]

# 读取数据中的标签列
# eye = point[['eye_x', 'eye_y', 'eye_w', 'eye_h', 'pipil_x', 'pupil_y', 'pupil_w', 'pupil_h']]
eye = point[['eye_x', 'eye_y', 'pipil_x', 'pupil_y', 'pupil_w', 'pupil_h']]
print(eye)


# clf = SGDR(loss='huber',penalty='l2',alpha=0.9,max_iter=1000)
clf = GBR(max_depth=10)
# clf = KNeighborsRegressor(n_neighbors=20, weights="distance", algorithm="ball_tree", leaf_size=50)
clf.fit(eye, world_x)
joblib.dump(clf, "../model/world_x.pkl")
print('X坐标预测得分：',clf.score(eye, world_x))

clf.fit(eye, world_y)
joblib.dump(clf, "../model/world_y.pkl")
print('Y坐标预测得分：',clf.score(eye, world_y))


# print('回归系数：',clf.coef_)
# print('偏差：',clf.intercept_)
