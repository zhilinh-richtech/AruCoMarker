import typing as _t

import numpy as np
import cv2

try:
    import pyrealsense2 as rs  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    rs = None  # type: ignore


class BaseCamera:
    def read(self) -> _t.Tuple[bool, np.ndarray]:  # (ok, frame_bgr)
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class RealSenseCamera(BaseCamera):
    def __init__(self, width: int = 1280, height: int = 800, fps: int = 30):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not available")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)

    def read(self) -> _t.Tuple[bool, np.ndarray]:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None  # type: ignore
        color_image = np.asanyarray(color_frame.get_data())
        return True, color_image

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


class OpenCVCamera(BaseCamera):
    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.cap = cv2.VideoCapture(device_index)
        # Best-effort settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        self.cap.set(cv2.CAP_PROP_FPS, float(fps))

    def read(self) -> _t.Tuple[bool, np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            return False, None  # type: ignore
        return True, frame

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


def create_camera(kind: str = "auto", width: int = 1280, height: int = 800, fps: int = 30, device: int = 0) -> BaseCamera:
    kind_norm = (kind or "auto").lower()
    if kind_norm == "auto":
        if rs is not None:
            return RealSenseCamera(width=width, height=height, fps=fps)
        return OpenCVCamera(device_index=device, width=width, height=height, fps=fps)
    if kind_norm == "realsense":
        return RealSenseCamera(width=width, height=height, fps=fps)
    if kind_norm in ("opencv", "uvc", "usb"):
        return OpenCVCamera(device_index=device, width=width, height=height, fps=fps)
    raise ValueError(f"Unknown camera kind: {kind}")


