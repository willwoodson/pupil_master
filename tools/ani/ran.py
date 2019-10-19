import cv2
import time
import numpy as np


height, weight = 1080, 1920
x, y, radius = 150, 150, 100
direction = [1,1]
data_video_record = "../../Data/data_video/ani.avi"

# 定义编解码器并创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(data_video_record, fourcc, 20.0, (weight, height))


def draw_ball(frame):
    global x, y, weight
    color_circle, thickness_circle = (250, 255, 255), 100
    color_line, thickness_line, length_line = (25, 255, 0), 10, 30

    cv2.circle(frame, (x, y), radius, color_circle, thickness=thickness_circle)

    cv2.line(frame, (x - length_line, y), (x + length_line, y), color_line, thickness=thickness_line)
    cv2.line(frame, (x, y - length_line), (x, y + length_line), color_line, thickness=thickness_line)


def movement():
    global height, weight, x, y, radius, direction
    move_x, move_y = 2, 5

    bound_l = 0 + radius + 50
    bound_r = weight - radius - 100
    bound_t = 0 + radius + 50
    bound_b = height - radius - 100

    x = x + move_x * direction[0]
    y = y + move_y * direction[1]

    if x > bound_r or x < bound_l:
        x = x - move_x * direction[0]
        direction[0] = -1 * direction[0]

    if y > bound_b or y < bound_t:
        y = y - move_y * direction[1]
        direction[1] = -1 * direction[1]

    time.sleep(0.01)


while True:
    frame = 0 * np.ones([height, weight, 3], np.uint8)  # 白色背景
    draw_ball(frame)
    movement()
    out.write(frame)

    if cv2.waitKey(10) == 27:
        break
    else:
        pass

    cv2.imshow("Demo", frame)

cv2.destroyAllWindows()