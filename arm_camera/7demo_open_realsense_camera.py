import pyrealsense2 as rs
import numpy as np
import cv2

# 创建 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()

# 配置彩色图像流（分辨率、帧率）
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动摄像头流
pipeline.start(config)

print("RealSense 摄像头已打开，按 ESC 退出")

try:
    while True:
        # 等待一帧图像
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # 转换为 NumPy 数组
        color_image = np.asanyarray(color_frame.get_data())

        # 显示图像
        cv2.imshow("RealSense Color Stream", color_image)

        # 按 ESC 退出
        if cv2.waitKey(1) == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("摄像头已关闭")
