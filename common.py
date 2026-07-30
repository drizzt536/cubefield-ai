class MissingPackage:
	def __init__(self, name) -> None:
		self.name = name

	def __getattr__(self, attr) -> None:
		raise AttributeError(f"required package `{self.name}` is missing")

# in case the external packages aren't actually required,
# don't crash until first access

try:
	import cv2
except ImportError:
	cv2 = MissingPackage("opencv-python")

try:
	import numpy as np
except ImportError:
	np = MissingPackage("numpy")

del MissingPackage

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ctypes
import time
import lzma
import os
import io

__all__ = (
	"ctypes", "time", "lzma", "io", "os", "cv2", "np",        # packages (and Path)
	"w", "h", "u", "r", "d", "l", "fps",                      # frame/video constants
	"Path", "runs_dir", "model_folder", "model_files",        # fs stuff
	"backup_history"
	"display", "downsample", "upsample", "downupsample",      # image processing
	"get_paths", "save_run", "load_runs", "count_datapoints", # training data
	"count_usages", "count_usable_datapoints",
	"clamp", "save_model", "load_model"                       # miscellaneous
)

runs_dir     = "runs"
model_folder = "models"

class _SaveBuffer(io.BytesIO):
	def __str__(self):
		return "model.keras"

from sys import platform

if platform == "win32":
	ctypes.windll.user32.SetProcessDPIAware()
	ctypes.windll.winmm.timeBeginPeriod(1)

	ctypes.windll.kernel32.SetPriorityClass(
		ctypes.windll.kernel32.GetCurrentProcess(),
		0 # high priority
	)
else:
	try:
		os.setpriority(os.PRIO_PROCESS, 0, -20)
	except PermissionError:
		# not running as root
		pass

del platform

w = 2560 # ctypes.windll.user32.GetSystemMetrics(0)
h = 1600 # ctypes.windll.user32.GetSystemMetrics(1)
u, r, d, l = 700, 600, 50, 600
fps = 60

backup_history = 10

model_files = (f"{model_folder}/game_model.keras.xz",) + tuple(
	f"{model_folder}/game_model.old{i if i > 1 else ''}.keras.xz"
	for i in range(1, backup_history + 1)
)

clamp = lambda x, min_, max_: min(max_, max(min_, x))

def display(img: np.ndarray) -> None:
	cv2.imshow("", img)

	while cv2.waitKey(16) != 0x71: # ord('q')
		pass

	cv2.destroyAllWindows()

def downsample(img):
	return cv2.resize(img, (160, 100), interpolation=cv2.INTER_AREA)

def upsample(img):
	return cv2.resize(img, (1360, 850), interpolation=cv2.INTER_NEAREST)

def downupsample(img):
	small = cv2.resize(img, (160, 100), interpolation=cv2.INTER_AREA)

	h, w = img.shape[:2]
	return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def get_paths(runs_dir=runs_dir, _file="*.npz") -> map:
	return Path(runs_dir).glob(_file)

def save_run(frames_data: list[tuple[np.ndarray, bool, bool, int]], runs_dir=runs_dir):
	if not frames_data:
		return

	Path(runs_dir).mkdir(exist_ok=True)

	run_id = int(time.time())
	path = Path(runs_dir) / f"{run_id}.npz"

	print(f"\rDownsampling {len(frames_data):,} frames\x1b[K", end="", flush=True)
	# frames = np.stack([downsample(f) for f, l, r, t in frames_data])  # (N, 100, 160) uint8
	old_thread_count = cv2.getNumThreads()
	cv2.setNumThreads(1)

	try:
		frames_only = [f for f, l, r, t in frames_data]
		frames = np.empty((len(frames_only), 100, 160), dtype=np.uint8)

		with ThreadPoolExecutor() as pool:
			for i, result in enumerate(pool.map(downsample, frames_only)):
				frames[i] = result
	finally:
		cv2.setNumThreads(old_thread_count)

	print(f"\rExtracting data from {len(frames_data):,} frames\x1b[K", end="", flush=True)
	left    = np.array([l for f, l, r, t in frames_data], dtype=np.bool)
	right   = np.array([r for f, l, r, t in frames_data], dtype=np.bool)
	elapsed = np.array([t for f, l, r, t in frames_data], dtype=np.uint32)

	print(f"\rSaving {len(frames_data):,} datapoints to {path.as_posix()}\x1b[K", end="", flush=True)
	np.savez_compressed(path, frames=frames, left=left, right=right, elapsed=elapsed)

	print(f"\rSaved {len(frames_data):,} datapoints to {path.as_posix()}\x1b[K")

def _pack_structured(frames, left, right, elapsed):
	h, w = frames.shape[1:]
	dtype = np.dtype([
		("frame", np.uint8, (h, w)),
		("left", np.bool),
		("right", np.bool),
		("elapsed", np.uint32),
	])

	data = np.empty(len(frames), dtype=dtype)
	data["frame"] = frames
	data["left"] = left
	data["right"] = right
	data["elapsed"] = elapsed
	return data

def load_runs(file="*.npz", runs_dir=runs_dir, raw: bool = False):
	all_frames, all_left, all_right, all_elapsed = [], [], [], []
	for path in get_paths(runs_dir, file):
		run = np.load(path)
		all_frames.append(run["frames"])
		all_left.append(run["left"])
		all_right.append(run["right"])
		all_elapsed.append(run["elapsed"])

	frames  = np.concatenate(all_frames)
	left    = np.concatenate(all_left)
	right   = np.concatenate(all_right)
	elapsed = np.concatenate(all_elapsed)

	if raw:
		return {
			"frames"  : frames,
			"left"    : left,
			"right"   : right,
			"elapsed" : elapsed,
		}

	return _pack_structured(frames, left, right, elapsed)

def count_datapoints(file="*.npz", runs_dir=runs_dir):
	return sum(len(np.load(path)["elapsed"]) for path in get_paths(runs_dir, file))

def count_usages(file="*.npz", runs_dir=runs_dir, turn_bias=0):
	"negative turn_bias biases towards doing nothing and positive biases towards turning."

	total = 0
	none  = 0
	left  = 0
	right = 0

	for path in get_paths(runs_dir, file):
		data = np.load(path)
		l = int(data["left"].sum())
		r = int(data["right"].sum())

		left  += l
		right += r
		none  += len(data["left"]) - l - r

	drop_p  = 1 - max(left, right) / none
	drop_p += ((turn_bias > 0) - drop_p)*abs(turn_bias)

	return {
		"total": none + left + right,
		"none" : none,
		"left" : left,
		"right": right,
		"drop" : drop_p
	}

def count_usable_datapoints(file="*.npz", runs_dir=runs_dir, turn_bias=0) -> int:
	"rough estimate. the actual value varies between training sessions"
	usages = count_usages(file, runs_dir, turn_bias)

	return usages["left"] + usages["right"] + round(usages["none"]*(1 - usages["drop"]))

def save_model(model, file=model_files[0]) -> None:
	buf = _SaveBuffer()
	model.save(buf)

	compressed = lzma.compress(buf.getvalue(), preset=9 | lzma.PRESET_EXTREME)

	with open(file, "wb") as f:
		f.write(compressed)

def load_model(file=model_files[0]):
	# don't import at the top of `common` since not all the programs need this.
	from keras.src.saving import saving_lib

	with open(file, "rb") as f:
		data = f.read()

	data = lzma.decompress(data)

	# this works on Keras 3.15.0
	# if it stops working, then just use a temp file instead
	return saving_lib._load_model_from_fileobj(
		io.BytesIO(data),
		custom_objects=None,
		compile=True,
		safe_mode=True,
	)
