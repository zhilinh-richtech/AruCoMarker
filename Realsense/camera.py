import typing as _t

import numpy as np
import cv2

try:
    import pyrealsense2 as rs  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    rs = None  # type: ignore

try:
    from pyorbbecsdk import Pipeline, Context, Config, OBFormat, OBSensorType  # type: ignore
    ORBBEC_AVAILABLE = True
    print("✓ pyorbbecsdk imported successfully")
except ImportError as e:  # pragma: no cover - optional dependency
    Pipeline = None  # type: ignore
    Context = None  # type: ignore
    Config = None  # type: ignore
    OBFormat = None  # type: ignore
    OBSensorType = None  # type: ignore
    ORBBEC_AVAILABLE = False
    print(f"⚠️ pyorbbecsdk not available: {e}")
except Exception as e:  # pragma: no cover - optional dependency
    Pipeline = None  # type: ignore
    Context = None  # type: ignore
    Config = None  # type: ignore
    OBFormat = None  # type: ignore
    OBSensorType = None  # type: ignore
    ORBBEC_AVAILABLE = False
    print(f"⚠️ pyorbbecsdk import error: {e}")

class BaseCamera:
    def read(self) -> _t.Tuple[bool, np.ndarray]:  # (ok, frame_bgr)
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
class OrbbecCamera(BaseCamera):
    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 30):
        if not ORBBEC_AVAILABLE:
            raise RuntimeError("pyorbbecsdk is not available")
        
        self.pipeline = Pipeline()
        self.config = Config()
        
        # Get available color stream profiles
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        profile_count = profile_list.get_count()
        
        # Find the desired profile (prefer MJPG for high resolution)
        selected_profile = None
        for i in range(profile_count):
            profile = profile_list.get_stream_profile_by_index(i)
            if profile.is_video_stream_profile():
                vp = profile.as_video_stream_profile()
                if (vp.get_width() == width and vp.get_height() == height and 
                    vp.get_format() == OBFormat.MJPG):
                    selected_profile = profile
                    break
        
        if selected_profile is None:
            # Fallback to first available profile
            selected_profile = profile_list.get_stream_profile_by_index(0)
        
        self.config.enable_stream(selected_profile)
        self.pipeline.start(self.config)
        
    def read(self) -> _t.Tuple[bool, np.ndarray]:
        try:
            frames = self.pipeline.wait_for_frames(3000)  # 3 second timeout
            if frames is None:
                return False, None  # type: ignore
            
            color_frame = frames.get_color_frame()
            if color_frame is None:
                return False, None  # type: ignore
            
            # Convert to numpy array
            color_data = np.asanyarray(color_frame.get_data())
            
            if color_data is None or color_data.size == 0:
                return False, None  # type: ignore
            
            # Handle MJPG format
            if color_frame.get_format() == OBFormat.MJPG:
                color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                if color_image is None:
                    return False, None  # type: ignore
            else:
                # Handle raw formats
                width = color_frame.get_width()
                height = color_frame.get_height()
                if len(color_data.shape) == 1:
                    expected_size = width * height * 3
                    if color_data.size == expected_size:
                        color_image = color_data.reshape((height, width, 3))
                        if color_frame.get_format() == OBFormat.RGB:
                            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                    else:
                        return False, None  # type: ignore
                else:
                    color_image = color_data
            
            return True, color_image
            
        except Exception:
            return False, None  # type: ignore
        
    def close(self) -> None:
        try:
            if hasattr(self, 'pipeline') and self.pipeline:
                self.pipeline.stop()
        except Exception:
            pass
        


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
        if ORBBEC_AVAILABLE:
            try:
                return OrbbecCamera(width=width, height=height, fps=fps)
            except Exception:
                print("Orbbec camera failed, falling back to RealSense")
                pass
        if rs is not None:
            return RealSenseCamera(width=width, height=height, fps=fps)
        return OpenCVCamera(device_index=device, width=width, height=height, fps=fps)
    if kind_norm == "realsense":
        return RealSenseCamera(width=width, height=height, fps=fps)
    if kind_norm == "orbbec":
        return OrbbecCamera(width=width, height=height, fps=fps)
    if kind_norm in ("opencv", "uvc", "usb"):
        return OpenCVCamera(device_index=device, width=width, height=height, fps=fps)
    raise ValueError(f"Unknown camera kind: {kind}")


