import cv2
import numpy as np
import shutil
import sys
import select
import termios
import tty
import time


def _get_term_size():
    size = shutil.get_terminal_size(fallback=(80, 24))
    return size.columns, max(5, size.lines - 2)


def _to_ascii_image(frame_bgr, max_w, max_h):
    aspect = 0.55
    h, w = frame_bgr.shape[:2]
    target_w = max(10, min(max_w, w))
    target_h = max(5, min(max_h, int(target_w * h / w * aspect)))
    if target_h < 5:
        target_h = 5
    resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    ramp = np.array(list(" .:-=+*#%@"), dtype="<U1")
    idx = (gray.astype(np.int32) * (len(ramp) - 1) // 255).astype(np.int32)
    lines = ["".join(ramp[row]) for row in idx]
    return "\n".join(lines)


def _to_braille_art(frame_bgr, max_w, max_h):
    """
    Render using Unicode braille (U+2800) with ordered dithering for smoother grayscale.
    Each character encodes a 2x4 dot cell, effectively 2 px wide x 4 px high.
    """
    h, w = frame_bgr.shape[:2]
    cell_cols = max(10, min(max_w, w // 2 if w >= 2 else 1))
    cell_rows = max(5, min(max_h, h // 4 if h >= 4 else 1))
    target_w = max(2, cell_cols * 2)
    target_h = max(4, cell_rows * 4)

    resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 4x2 ordered dither threshold matrix (values in (0,1))
    thresh = np.array([
        [0.125, 0.625],
        [0.375, 0.875],
        [0.0625,0.5625],
        [0.3125,0.8125],
    ], dtype=np.float32)

    lines = []
    for r in range(0, target_h, 4):
        chars = []
        for c in range(0, target_w, 2):
            bits = 0
            # Compare each pixel to its threshold to decide dot state
            g00 = gray[r + 0, c + 0] > thresh[0, 0]
            g10 = gray[r + 1, c + 0] > thresh[1, 0]
            g20 = gray[r + 2, c + 0] > thresh[2, 0]
            g30 = gray[r + 3, c + 0] > thresh[3, 0]
            g01 = gray[r + 0, c + 1] > thresh[0, 1]
            g11 = gray[r + 1, c + 1] > thresh[1, 1]
            g21 = gray[r + 2, c + 1] > thresh[2, 1]
            g31 = gray[r + 3, c + 1] > thresh[3, 1]

            bits |= (1 if g00 else 0) << 0
            bits |= (1 if g10 else 0) << 1
            bits |= (1 if g20 else 0) << 2
            bits |= (1 if g30 else 0) << 6
            bits |= (1 if g01 else 0) << 3
            bits |= (1 if g11 else 0) << 4
            bits |= (1 if g21 else 0) << 5
            bits |= (1 if g31 else 0) << 7

            chars.append(chr(0x2800 + bits) if bits != 0 else ' ')
        lines.append("".join(chars))
    return "\n".join(lines)


class _TerminalUI:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        try:
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        except Exception:
            self.old = None
        sys.stdout.write("\x1b[2J\x1b[H\x1b[?25l")
        sys.stdout.flush()

    def restore(self):
        if self.old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            except Exception:
                pass
        sys.stdout.write("\x1b[?25h\n")
        sys.stdout.flush()

    def draw_ascii(self, frame_bgr, status_text=""):
        cols, rows = _get_term_size()
        art = _to_ascii_image(frame_bgr, cols, rows)
        sys.stdout.write("\x1b[H")
        sys.stdout.write(art)
        if status_text:
            sys.stdout.write("\n" + status_text)
        sys.stdout.flush()

    def draw_text(self, art_text: str, status_text: str = ""):
        sys.stdout.write("\x1b[H")
        sys.stdout.write(art_text)
        if status_text:
            sys.stdout.write("\n" + status_text)
        sys.stdout.flush()

    def read_key(self):
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch:
                    return ch
        except Exception:
            return None
        return None


class Display:
    """
    Unified display helper for three modes:
      - gui: OpenCV window
      - ascii: terminal ASCII art with non-blocking key input
      - headless: periodic status line, non-blocking key input
    """

    def __init__(self, mode: str, window_name: str = "RealSense"):
        self.mode = mode
        self.window_name = window_name
        self._term = None
        self._last_key = None
        self._last_status_time = 0.0
        if self.mode in ("ascii", "ascii_hi", "ascii_braille", "headless"):
            self._term = _TerminalUI()

    def update(self, frame_bgr, status_text: str = ""):
        """
        Render the frame/status for the current mode and return a recent key if pressed.
        Returns: a single-character string or None
        """
        self._last_key = None
        if self.mode == "gui":
            cv2.imshow(self.window_name, frame_bgr)
            k = cv2.waitKey(1) & 0xFF
            if k != 255:
                try:
                    self._last_key = chr(k)
                except Exception:
                    self._last_key = None
            return self._last_key

        if self.mode == "ascii":
            self._term.draw_ascii(frame_bgr, status_text)
            self._last_key = self._term.read_key()
            time.sleep(1/15)
            return self._last_key

        if self.mode in ("ascii_hi", "ascii_braille"):
            cols, rows = _get_term_size()
            art = _to_braille_art(frame_bgr, cols, rows)
            self._term.draw_text(art, status_text)
            self._last_key = self._term.read_key()
            time.sleep(1/15)
            return self._last_key

        # headless
        now = time.time()
        if now - self._last_status_time >= 0.5:
            sys.stdout.write("\r" + status_text + "   ")
            sys.stdout.flush()
            self._last_status_time = now
        self._last_key = self._term.read_key() if self._term is not None else None
        time.sleep(1/30)
        return self._last_key

    def close(self):
        if self.mode == "gui":
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if self._term is not None:
            self._term.restore()



def draw_axes_ascii_friendly(
    img_bgr,
    K,
    dist,
    R,
    t,
    axis_len_m: float = 0.08,
    thickness: int = 4,
):
    """
    Draw a high-contrast white axis overlay that survives ASCII/Braille conversion.
    Uses thicker white lines for X (red), Y (green), Z (blue) equivalents but in white for clarity.
    """
    # Define 3D points for origin and axes endpoints
    axis = np.float32([
        [0, 0, 0],
        [axis_len_m, 0, 0],
        [0, axis_len_m, 0],
        [0, 0, axis_len_m],
    ]).reshape(-1, 3)

    # Convert R to rvec if needed
    if R.shape == (3, 3):
        rvec, _ = cv2.Rodrigues(R)
    else:
        rvec = R
    tvec = t.reshape(3, 1)

    pts_2d, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    pts_2d = np.int32(pts_2d.reshape(-1, 2))

    o = tuple(pts_2d[0])
    x = tuple(pts_2d[1])
    y = tuple(pts_2d[2])
    z = tuple(pts_2d[3])

    # Draw thick white lines to ensure visibility after grayscale/dither
    cv2.line(img_bgr, o, x, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    cv2.line(img_bgr, o, y, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    cv2.line(img_bgr, o, z, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    # Draw small white circles at endpoints
    for p in (o, x, y, z):
        cv2.circle(img_bgr, p, max(1, thickness // 2), (255, 255, 255), -1, lineType=cv2.LINE_AA)

