import time
# do this before importing the rest of the modules since imports can be slow
startup_start = time.perf_counter_ns()

from sys import argv

from common import (
	np, ctypes, cv2, Path,
	save_run, downsample,
	model_folder, model_files,
	load_model,
	fps as _fps, w, h, u, r, d, l
)

user32           = ctypes.windll.user32
GetAsyncKeyState = user32.GetAsyncKeyState

# score borders
su, sr, sd, sl = 50, 2250, 1470, 50

vis_title   = "AI Brain"
frame_shape = (100, 160)

fps  = None
vis  = True
save = True
collect = False
auto_retry = False
model_file = None
vis_signed = None
vis_size   = None

for arg in argv[1:]:
	match arg:
		case "--nosave":
			if not save:
				raise ValueError("`--nosave` given twice")

			save = False
		case "--collect":
			if collect:
				raise ValueError("`--collect` given twice")

			collect = True
		case "--autoretry":
			if auto_retry:
				raise ValueError("`--autoretry` given twice")

			auto_retry = True
		case _s if _s.startswith("--fps="):
			if fps is not None:
				raise ValueError("`--fps` given twice")

			try:
				fps = int(arg[6:])

				if fps < 1:
					raise ValueError
			except ValueError:
				raise ValueError("`--fps=FPS` given a non integer, or an integer less than 1")
		case "--vis=off":
			if vis_signed is True:
				raise ValueError("`--vis=signed` and `--vis=off` cannot be given together")

			if vis_signed is False:
				raise ValueError("`--vis=unsigned` and `--vis=off` cannot be given together")

			if not vis:
				raise ValueError("`--vis=off` given twice")

			vis = False
		case "--vis=signed":
			if vis_signed is True:
				raise ValueError("`--vis=signed` given twice")

			if vis_signed is False:
				raise ValueError("`--vis=signed` and `--vis=unsigned` cannot be given together")

			if not vis:
				raise ValueError("`--vis=signed` and `--vis=off` cannot be given together")

			vis_signed = True
		case "--vis=unsigned":
			if vis_signed is True:
				raise ValueError("`--vis=signed` and `--vis=unsigned` cannot be given together")

			if vis_signed is False:
				raise ValueError("`--vis=unsigned` given twice")

			if not vis:
				raise ValueError("`--vis=unsigned` and `--vis=off` cannot be given together")

			vis_signed = False
		case _s if _s.startswith("--vis-size="):
			if vis_size is not None:
				raise ValueError("`--vis-size` given twice")

			try:
				vis_size = float(arg[11:])
			except ValueError:
				raise ValueError("`--vis-size=VAL` given a non floating point value")

			if abs(vis_size - round(vis_size)) < 1e-4:
				vis_size = round(vis_size)

			if vis_size == 0:
				# copy/pasted from the `--vis=off` branch
				if vis_signed is True:
					raise ValueError("`--vis=signed` and `--vis=off` cannot be given together")

				if vis_signed is False:
					raise ValueError("`--vis=unsigned` and `--vis=off` cannot be given together")

				if not vis:
					raise ValueError("`--vis=off` given twice")

				vis = False
		case "--help" | "-h":
			print(
				"usage: python play.py [FLAGS | FILE]"
				"\n"
				"\noptions:"
				"\n    --help, -h      print this message and exit"
				"\n    --vis=off       don't display the AI pixel heatmap"
				"\n    --vis=signed    display signed heatmap. green is positive and red is negative."
				"\n                    only takes effect if `--vis=off` was not also given."
				"\n    --vis=unsigned  noop. this is the default."
				"\n    --autoretry     attempt to auto detect and restart when the play dies"
				"\n                    in the neon green and pink areas, it might not detect immediately"
				"\n                    due to clouds. it waits 2 seconds before declaring death"
				"\n    --vis-size=M    multiplier for visualizer size. default is 6. can be an int or float."
				"\n                    any higher than 7, and it will probably have issues."
				"\n                    --vis-size=0 is an alias for --vis=off"
				"\n    --nosave        don't collect or save data during manual takeover (or with --collect)."
				"\n    --collect       start with the AI paused and cut the first and last few seconds of"
				"\n                    the run. partially overridden by --nosave. also puts black borders"
				"\n                    on the screen to cover the game area outside of the AIs visibility"
				"\n                    --autoretry is ignored if --collect is given."
				f"\n    --fps=FPS       change the FPS value. default is '{_fps}'. setting the FPS higher than"
				f"\n                    what it naturally caps at causes console printouts to happen slower."
				f"\n    FILE            model file. default is '{model_files[0]}' if not given."
				"\n"
				"\nAI controls:"
				"\n    escape    stop the current run, or exit the program if a run is not active"
				"\n    space     start a run, if the prompt is visible"
				"\n    s         change signedness of the visualizer"
				"\n    t         take over or return control to the AI. collects training data when taken over"
				"\n"
				"\ngame controls:"
				"\n    p         pause the game. doesn't interact well with `--autoretry`"
				"\n    q         change game quality. it is recommended to leave quality on high."
				"\n    left      move left"
				"\n    right     move right"
				"\n"
				"\nkey-value flags cannot be given as `--key val` and must be given as `--key=val`."
				"\nrequires numpy, opencv-python, dxcam, tensorflow, pytesseract, and pydirectinput."
				"\nOnly works on Windows. probably only works if the primary monitor is 2560x1600."
				"\nthe game is assumed to be in fullscreen on the primary monitor."
			)

			exit(0)
		case _:
			if arg.startswith("-"):
				raise ValueError(f"unrecognized flag: '{arg}'")

			if model_file is not None:
				raise ValueError("multiple model files cannot be passed")

			if not arg:
				raise ValueError("model file cannot be an empty string")

if vis_size is None:
	vis_size = 6

if fps is None:
	fps = _fps

del _fps

vis_shape = tuple(round(x*vis_size) for x in frame_shape)

if vis_signed is None:
	vis_signed = False

if model_file is None:
	model_file = model_files[0]

if collect:
	auto_retry = False

# import as much as possible after parsing flags, in case --help was passed
from shutil import copy as file_copy
from collections import deque
import tensorflow as tf
import pydirectinput
import pytesseract
import subprocess
import threading
import dxcam
import queue

pydirectinput.PAUSE = 0.0

Path(model_folder).mkdir(exist_ok=True)

model = load_model(model_file)
action_labels = ["none", "left", "right"]

# the heatmap needs raw logits without softmax
logit_model = tf.keras.models.clone_model(model)
logit_model.layers[-1].activation = tf.keras.activations.linear
logit_model.set_weights(model.get_weights())

@tf.function
def get_fast_action_and_saliency(image_tensor, time_tensor):
	# Use real model to play the game
	probabilities = model([image_tensor, time_tensor], training=False)
	chosen_action = tf.cast(tf.argmax(probabilities[0]), tf.int32)

	with tf.GradientTape() as tape:
		tape.watch(image_tensor)
		logits = logit_model([image_tensor, time_tensor], training=False)
		target_score = logits[0, chosen_action]

	image_gradients = tape.gradient(target_score, image_tensor)
	return chosen_action, image_gradients

@tf.function
def fast_predict(img, time_val):
	return model([img, time_val], training=False)[0]

input_queue  = queue.Queue(maxsize=1)
output_queue = queue.Queue(maxsize=1)

def background_visualizer():
	while True:
		data = input_queue.get()
		if data is None: # poison pill to shut down
			break

		frame_input, elapsed_input = data

		frame_input   = tf.convert_to_tensor(frame_input,   dtype=tf.float32)
		elapsed_input = tf.convert_to_tensor(elapsed_input, dtype=tf.float32)

		action, img_grads = get_fast_action_and_saliency(frame_input, elapsed_input)
		action = action_labels[action.numpy()]

		display = img_grads.numpy().squeeze() # raw saliency

		if vis_signed:
			display /= np.abs(display).max() + 1e-8 # normalized saliency
			display  = np.stack([
				np.zeros_like(display, dtype=np.uint8), # B
				np.uint8(  display .clip(0, 1) * 255),  # G
				np.uint8((-display).clip(0, 1) * 255),  # R
			], axis=-1)
		else:
			display  = np.abs(display)      # absolute saliency
			display /= display.max() + 1e-8 # normalized saliency
			display  = np.uint8(255 * display)
			display  = cv2.applyColorMap(display, cv2.COLORMAP_HOT)

		cv2.putText(
			display, f"Action: {action}",
			(5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
		)

		display = cv2.resize(
			display,
			vis_shape[::-1],
			interpolation=cv2.INTER_NEAREST
		)

		try:
			output_queue.put_nowait(display)
		except queue.Full:
			_ = output_queue.get()
			output_queue.put(display)

		input_queue.task_done()

def main(res: int = 1) -> bool:
	global vis_signed

	if res == 0:
		res = 1

	cam = dxcam.create(backend="winrt", output_color="BGRA")
	pause_frames = deque()

	if vis:
		# idk why AUTOSIZE is required
		cv2.namedWindow(vis_title, cv2.WINDOW_AUTOSIZE)
		cv2.setWindowProperty(vis_title, cv2.WND_PROP_TOPMOST, 1)

		placeholder = np.zeros(frame_shape + (3,), dtype=np.uint8)
		cv2.putText(
			placeholder, "waiting for input",
			(5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
		)
		placeholder = cv2.resize(
			placeholder,
			vis_shape[::-1],
			interpolation=cv2.INTER_NEAREST
		)
		cv2.imshow(vis_title, np.zeros( (10, 10, 1) ))
		cv2.waitKey(1)

		hwnd  = user32.FindWindowW(None, vis_title)
		style = user32.GetWindowLongW(hwnd, -16) & 0xff3bffff
		user32.SetWindowLongW(hwnd, -16, style)
		user32.SetWindowPos(hwnd, None, w - placeholder.shape[1], 0, 0, 0, 0x25)
		cv2.imshow(vis_title, placeholder)

		del placeholder, hwnd, style

	if res == 2:
		pydirectinput.keyDown("space")
		time.sleep(0.1)
		pydirectinput.keyUp("space")
	else:
		print("waiting for spacebar \x1b7down or esc", end="", flush=True)

		while GetAsyncKeyState(0x20) >= 0: # space
			if vis: cv2.waitKey(1)

			if GetAsyncKeyState(0x1b) < 0: # escape
				print()
				cam.release()
				if vis: cv2.destroyAllWindows()
				return 0

		print("\x1b8up\x1b[K", end="", flush=True)
		while GetAsyncKeyState(0x20) < 0:
			if vis:
				cv2.waitKey(1)

	if vis:
		vis_thread = threading.Thread(target=background_visualizer, daemon=True)
		vis_thread.start()

	cam.start(target_fps=fps, video_mode=True)

	print(f"\relapsed: \x1b70s, r=0.00, fps=0.00\x1b[K", end="", flush=True)

	prev_vis_signed_toggle = False
	prev_pause_toggle      = False

	prev_elapsed   = -1
	current_action = 0 # 0=none, 1=left, 2=right
	frozen_frames  = 0
	prev_score     = None

	paused     = collect
	game_start = time.perf_counter()
	fps_start  = game_start
	frame_i    = 0
	ratio      = 0
	res        = 1


	while GetAsyncKeyState(0x1b) >= 0: # while escape not pressed
		frame = cam.get_latest_frame()

		# only used if it is paused
		user_left  = GetAsyncKeyState(0x25)
		user_right = GetAsyncKeyState(0x27)

		score = frame[su : h - sd, sl : w - sr]
		frame = frame[ u : h -  d,  l : w -  r]
		frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

		frame_input = cv2.resize(frame, frame_shape[::-1], interpolation=cv2.INTER_AREA)
		frame_input = (frame_input.astype(np.float32) / 255)[None, ..., None]

		elapsed_seconds = int(time.perf_counter() - game_start)
		elapsed_input = np.float32([[elapsed_seconds / 300]])

		logits = fast_predict(frame_input, elapsed_input)
		action = int(np.argmax(logits))

		pause_toggle = GetAsyncKeyState(0x54) < 0 # ord('T')
		if pause_toggle and not prev_pause_toggle:
			# toggle key rising edge

			paused = not paused
			print(
				f"\r[AI {"PAUSED" if paused else "RESUMED"}]\x1b[K\n"
				f"elapsed: \x1b7{elapsed_seconds}s, r={ratio:.2f}, fps={fps*ratio:.2f}\x1b[K",
				end="", flush=True
			)

			if paused:
				pydirectinput.keyUp("left")
				pydirectinput.keyUp("right")
				current_action = 0

		prev_pause_toggle = pause_toggle

		if paused:
			if user_left < 0 and user_right < 0:
				user_left = user_right = 0 # not less than 0

			if save:
				pause_frames.append((frame, user_left < 0, user_right < 0, elapsed_seconds))
		elif action != current_action:
			if current_action == 1:
				pydirectinput.keyUp("left")
			elif current_action == 2:
				pydirectinput.keyUp("right")

			if action == 1:
				pydirectinput.keyDown("left")
			elif action == 2:
				pydirectinput.keyDown("right")

			current_action = action

		if elapsed_seconds != prev_elapsed:
			prev_elapsed = elapsed_seconds

			# update the elapsed time as often as possible
			print(
				f"\x1b8{elapsed_seconds}s, r={ratio:.2f}, fps={fps * ratio:.2f}\x1b[K",
				end="", flush=True
			)

		frame_i += 1
		if frame_i == fps:
			frame_i = 0
			fps_end = time.perf_counter_ns()
			ratio   = 1e9 / (fps_end - fps_start)
			fps_start = fps_end

			# update the ratio and fps as often as possible
			print(
				f"\x1b8{elapsed_seconds}s, r={ratio:.2f}, fps={fps * ratio:.2f}\x1b[K",
				end="", flush=True
			)

		if vis:
			vis_signed_toggle = GetAsyncKeyState(0x53) < 0 # ord('S')

			if vis_signed_toggle and not prev_vis_signed_toggle:
				# I don't think I care if this causes race conditions
				# it doesn't appear to cause issues.
				vis_signed = not vis_signed

			prev_vis_signed_toggle = vis_signed_toggle

			try:
				input_queue.put_nowait((frame_input, elapsed_input))
			except queue.Full:
				pass

			# check if the thread finished a heatmap
			try:
				finished_display = output_queue.get_nowait()
				cv2.imshow(vis_title, finished_display)
				cv2.waitKey(1)
			except queue.Empty:
				pass

		if auto_retry:
			score = cv2.cvtColor(score, cv2.COLOR_BGRA2GRAY)

			if prev_score is None:
				prev_score = score
				continue

			diff = cv2.absdiff(score, prev_score)

			if cv2.countNonZero(diff) != 0:
				frozen_frames = 0
				prev_score = score
				continue

			frozen_frames += 1

			if frozen_frames >= fps << 1:
				print(f"\n[DEATH DETECTED] Restarting...")

				new_score = int(pytesseract.image_to_string(
					score,
					config=(
						"--psm 13"
						" -c tessedit_char_whitelist=0123456789"
						" -c classify_bln_numeric_mode=1"
					)
				))

				old_pb_paths = tuple(Path(model_folder).glob("pb-model-*.keras"))

				if len(old_pb_paths) > 1:
					# delete all the stale PB files if there are any
					best_path = max(old_pb_paths, key=lambda p: int(p.name[9:-6]))

					for path in old_pb_paths:
						if path != best_path:
							path.unlink()

					old_pb_score = int(best_path.name[9:-6])
				elif old_pb_paths:
					old_pb_score = int(old_pb_paths[0].name[9:-6])
				else:
					old_pb_score = 0

				pb = ""
				if new_score > old_pb_score:
					pb = " (PB)"

					if old_pb_paths:
						old_pb_paths[0].unlink()

					file_copy(model_file, f"{model_folder}/pb-model-{new_score}.keras")

				print(f"score{pb}: {new_score:,}", end="")

				res = 2
				break

	print("\n", end="")
	cam.stop()
	cam.release()
	pydirectinput.keyUp("left")
	pydirectinput.keyUp("right")

	if vis:
		input_queue.put(None)
		vis_thread.join(timeout=1)
		cv2.destroyAllWindows()

	if collect:
		# skip some stuff at the start and end of the run
		try:
			for _ in range(fps):
				pause_frames.popleft()

			for _ in range(fps << 1):
				pause_frames.pop()
		except IndexError:
			# not enough datapoints
			pass

		if not pause_frames:
			print(
				"run wasn't long enough to produce any datapoints"
				if save else
				"run is not being saved due to `--nosave`."
			)

	if save:
		save_run(pause_frames)

	print()
	return res

if collect:
	print("starting border subprocess")
	blackout = subprocess.Popen([
		"python",
		"border-blackout.py",
		str(u),
		str(r),
		str(d),
		str(l),
	])

	# wait so the border windows are created first so if `--vis=off` wasn't given,
	# the heatmap window will be create after, putting it on top.
	if vis:
		print("waiting 1s")
		time.sleep(1)

prgm_start = time.perf_counter_ns()

print(f"startup time: {(prgm_start - startup_start) / 1e9:.2f}s")

res  = 1
runs = 0
while res != 0:
	res = main(res)
	runs += 1

prgm_end = time.perf_counter_ns()

print(
	f"\ntotal runs: {runs - 1}" # last run exits early but still increments the counter
	f"\ntotal elapsed time: {(prgm_end - prgm_start) / 1e9:.2f}s"
	f"\nexiting"
)

if collect:
	blackout.kill()
	blackout.wait()
