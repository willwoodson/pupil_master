import sys

import pygame
from pygame.locals import *


def play_ball():
    pygame.init()

    # 窗口大小
    window_size = (width, height) = (1920, 1080)

    # 小球运行偏移量[水平，垂直]，值越大，移动越快
    speed = [10, 10]

    # 窗口背景色RGB值
    color_black = (0, 0, 139)

    # 设置窗口模式
    screen = pygame.display.set_mode(window_size)

    # 设置窗口标题
    pygame.display.set_caption('运动的小球')

    # 加载小球图片
    ball_image = pygame.image.load('../../images/animation-200x200.png')

    # 获取小球图片的区域开状
    ball_rect = ball_image.get_rect()

    frames_per_sec = 10
    fps_clock = pygame.time.Clock()

    while True:

        # 退出事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 使小球移动，速度由speed变量控制
        ball_rect = ball_rect.move(speed)

        # 当小球运动出窗口时，重新设置偏移量
        if (ball_rect.left < 0) or (ball_rect.right > width):
            speed[0] = - speed[0]
        if (ball_rect.top < 0) or (ball_rect.bottom > height):
            speed[1] = - speed[1]

        # 填充窗口背景
        screen.fill(color_black)

        # 在背景Surface上绘制 小球
        screen.blit(ball_image, ball_rect)

        # 更新窗口内容
        pygame.display.update()

        fps_clock.tick(frames_per_sec)


if __name__ == '__main__':
    play_ball()
