import pyrealsense2 as rs

# 创建管道并启用流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 获取流配置
profile = pipeline.get_active_profile()
depth_profile = profile.get_stream(rs.stream.depth)
color_profile = profile.get_stream(rs.stream.color)


depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

print("=== Depth Intrinsics ===")
print(depth_intrinsics)

print("\n=== Color Intrinsics ===")
print(color_intrinsics)

extrinsics = depth_profile.get_extrinsics_to(color_profile)
print("\n=== Extrinsics (Depth → Color) ===")
print(extrinsics)

pipeline.stop()

