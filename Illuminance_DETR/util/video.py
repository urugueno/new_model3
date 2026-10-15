"""
ブラウザで再生できる mp4 を書くための VideoWriter。

なぜ cv2.VideoWriter を使わないか:
  従来は fourcc 'mp4v' = MPEG-4 Part 2 で書いていたが、これは拡張子が .mp4 でも
  Chrome / Firefox / Edge / Safari のいずれもデコードできず、Chromium ベースの
  VS Code 拡張（mpeg-4 preview 等）でも再生できない。
  ブラウザ再生には H.264 (AVC) + yuv420p が必要。
  しかし .venv の OpenCV に同梱された libavcodec には使える H.264 エンコーダが
  無い（h264_v4l2m2m のみ。libx264 も libopenh264 も入っていない）ため、
  fourcc を 'avc1' に変えても出力できない。
  そこで生の BGR フレームを ffmpeg にパイプして libx264 でエンコードする。

ffmpeg は imageio-ffmpeg が .venv 内に同梱するバイナリを使うので、
システムへの ffmpeg インストールは不要（slurm の計算ノードでも .venv さえ
共有されていれば動く）。

使い方は cv2.VideoWriter と同じ:

    writer = H264VideoWriter(path, fps, (width, height))
    if not writer.isOpened():
        ...
    try:
        for frame in frames:      # BGR uint8
            writer.write(frame)
    finally:
        writer.release()

size に None を渡すと最初のフレームの形状から決める。matplotlib の figure を
書き出す場合、実サイズは figsize x figure.dpi で決まるので、サイズを決め打ちせず
None にしておく方が安全。
"""
import os
import shutil
import subprocess

import cv2
import numpy as np

# H.264 の品質・速度。crf は小さいほど高画質（18〜28 が実用域）。
DEFAULT_CRF = 23
DEFAULT_PRESET = "veryfast"

_INSTALL_HINT = (
    "ffmpeg が見つかりません。以下のいずれかで用意してください:\n"
    "  uv pip install imageio-ffmpeg      （推奨・.venv 内にバイナリを同梱）\n"
    "  sudo apt install ffmpeg            （システムに入れる場合）"
)


def find_ffmpeg() -> str:
    """
    使用する ffmpeg の実行パスを返す。
    imageio-ffmpeg の同梱バイナリを優先し、無ければ PATH 上の ffmpeg を探す。
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    exe = shutil.which("ffmpeg")
    if exe:
        return exe

    raise RuntimeError(_INSTALL_HINT)


class H264VideoWriter:
    """cv2.VideoWriter の差し替え。H.264 / yuv420p / +faststart の mp4 を書く。"""

    def __init__(self, path, fps, size=None, crf=DEFAULT_CRF, preset=DEFAULT_PRESET):
        """
        path : 出力先（.mp4）
        fps  : フレームレート
        size : (width, height)。None なら最初の write() のフレームから決める
        """
        self.path = path
        self.fps = float(fps)
        self.crf = crf
        self.preset = preset

        self._proc = None
        self._size = tuple(size) if size is not None else None
        self._frames = 0
        self._resize_warned = False
        self._failed = False

        try:
            self._ffmpeg = find_ffmpeg()
        except RuntimeError as e:
            print(f"[video] [error] {e}")
            self._ffmpeg = None
            self._failed = True
            return

        # size が決まっているならこの場で起動し、cv2.VideoWriter と同じく
        # isOpened() で開けたかどうかを判定できるようにする。
        if self._size is not None:
            self._start()

    def _start(self):
        width, height = self._size

        out_dir = os.path.dirname(os.path.abspath(self.path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        cmd = [
            self._ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-framerate", f"{self.fps}",
            "-i", "-",
            "-an",
        ]
        # yuv420p は偶数サイズが必要。奇数のときだけ 1px 分パディングする。
        if width % 2 or height % 2:
            cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        cmd += [
            "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",   # moov を先頭に置きストリーミング再生可能にする
            self.path,
        ]

        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except OSError as e:
            print(f"[video] [error] ffmpeg を起動できませんでした: {e}")
            self._failed = True

    def isOpened(self) -> bool:
        """size 未確定（遅延起動待ち）も「開いている」とみなす。"""
        if self._failed:
            return False
        return self._size is None or self._proc is not None

    def write(self, frame):
        if self._failed:
            return

        frame = np.ascontiguousarray(frame)

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"[video] BGR 3ch のフレームが必要です: shape={frame.shape}")
        if frame.dtype != np.uint8:
            raise ValueError(
                f"[video] uint8 のフレームが必要です: dtype={frame.dtype}")

        if self._size is None:
            h, w = frame.shape[:2]
            self._size = (w, h)
            self._start()
            if self._failed:
                return

        width, height = self._size
        if frame.shape[0] != height or frame.shape[1] != width:
            # cv2.VideoWriter はサイズ違いのフレームを黙って捨てるが、生の
            # パイプでは映像が壊れる。コマ落ちさせるよりリサイズして残す。
            if not self._resize_warned:
                print(f"[video] [warn] フレームサイズが宣言と異なります "
                      f"({frame.shape[1]}x{frame.shape[0]} != {width}x{height})。"
                      f"リサイズして書き込みます（以降は警告しません）")
                self._resize_warned = True
            frame = cv2.resize(frame, (width, height))

        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            self._failed = True
            self._report_ffmpeg_error("書き込み中に ffmpeg が終了しました")
            return

        self._frames += 1

    def _report_ffmpeg_error(self, headline):
        err = b""
        if self._proc is not None and self._proc.stderr is not None:
            try:
                err = self._proc.stderr.read() or b""
            except Exception:
                pass
        print(f"[video] [error] {headline}: {self.path}")
        if err:
            print(err.decode("utf-8", "replace").rstrip())

    def release(self):
        if self._proc is None:
            return

        proc, self._proc = self._proc, None
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

        err = b""
        if proc.stderr is not None:
            try:
                err = proc.stderr.read() or b""
            except Exception:
                pass
        proc.wait()

        if proc.returncode != 0:
            print(f"[video] [error] ffmpeg が失敗しました "
                  f"(exit={proc.returncode}): {self.path}")
            if err:
                print(err.decode("utf-8", "replace").rstrip())
            self._failed = True
        elif err:
            print(err.decode("utf-8", "replace").rstrip())

    # with 文でも使えるようにする（例外時の release 漏れ防止）
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def __del__(self):
        # release() を呼び忘れても ffmpeg プロセスを残さない
        try:
            self.release()
        except Exception:
            pass
