"""
commands:
	space/enter: pick new random frame from a random file

	left:       go back 1 frame
	shift+left: go back 10 frames
	ctrl+left:  go back 1 second

	right:       advance 1 frame
	shift+right: advance 10 frames
	ctrl+right:  advance 1 second

	home: go to the start of the run
	end:  go to the end of the run

	f: toggle fullscreen
	s: toggle signedness of heatmaps
	q: quit
"""

from sys import argv

if len(argv) > 1 and argv[1] in ("--help", "-h"):
	print("usage: python review-saliency.py [--help | -h | MODEL_FILE]")
	print(__doc__[:-1])
	exit(0)

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from common import np, get_paths, load_runs, load_model, model_files, clamp, fps

model = load_model(argv[1] if len(argv) > 1 else model_files[0])
action_labels = ["none", "left", "right"]

logit_model = tf.keras.models.clone_model(model)
logit_model.layers[-1].activation = tf.keras.activations.linear
logit_model.set_weights(model.get_weights())

current_data = None
current_index = 0

def pick_random_frame():
	global current_data, current_index

	file = np.random.choice(list(get_paths())).name
	current_data = load_runs(file)

	current_index = np.random.randint(0, len(current_data))
	return current_data[current_index]

def compute_frame(frame_data):
	global current_index
	print(f"\n--- Analyzing Frame {current_index} ---")

	single_frame = frame_data[0][None, ..., None]
	single_time  = (frame_data[3].astype(np.float32) / 300)[None, None]

	image_tensor = tf.convert_to_tensor(single_frame, dtype=tf.float32)
	time_tensor  = tf.convert_to_tensor(single_time, dtype=tf.float32)

	saliency_maps = []
	p = model([image_tensor, time_tensor], training=False)[0].numpy()

	for action_index in range(3):
		with tf.GradientTape() as tape:
			tape.watch(image_tensor)
			tape.watch(time_tensor)

			logits = logit_model([image_tensor, time_tensor], training=False)
			target_score = logits[0, action_index]

		image_gradients, time_gradients = tape.gradient(
			target_score,
			[image_tensor, time_tensor]
		)

		raw_grads = image_gradients.numpy().squeeze()
		input_img = image_tensor.numpy().squeeze()
		saliency  = raw_grads * input_img

		# normalize to [-1, +1]
		saliency /= np.abs(saliency).max() + 1e-8
		saliency_maps.append(saliency)

		d_time = time_gradients.numpy()[0, 0]
		print(f"Action {action_index} ({action_labels[action_index]:<5}) -> dP/dt: {d_time:+.6f}, P={target_score:+9.3f}, p={p[action_index]:.6f}")

	return single_frame.squeeze(), saliency_maps, p

print("NOTE: p is the probability, and P is the raw preference value")

rbg_cmap = LinearSegmentedColormap.from_list("RedBlackGreen", ["red", "black", "lime"])

plt.style.use("dark_background")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes_flat = axes.flatten()

initial_data = pick_random_frame()

show_signed = True
current_frame_img, current_maps, current_p = compute_frame(initial_data)

im_objects = []

axes_flat[0].set_title("Input Frame\n(Press ENTER for new frame)")
im = axes_flat[0].imshow(current_frame_img, cmap="gray")
axes_flat[0].axis("off")
im_objects.append(im)

for idx in range(3):
	ax = axes_flat[idx + 1]
	ax.set_title(f"Action {idx}: {action_labels[idx]}, p={current_p[idx]:.4f}")
	im = ax.imshow(current_maps[idx], cmap=rbg_cmap, vmin=-1, vmax=1)
	ax.axis("off")
	im_objects.append(im)

cbar = fig.colorbar(im_objects[1], ax=axes, fraction=0.03, pad=0.04)

def update_display():
	im_objects[0].set_data(current_frame_img)

	cmap = rbg_cmap if show_signed else "hot"
	vmin, vmax = (-1, 1) if show_signed else (0, 1)

	for idx in range(3):
		display_map = current_maps[idx] if show_signed else np.abs(current_maps[idx])

		im = im_objects[idx + 1]
		im.set_data(display_map)
		im.set_cmap(cmap)
		im.set_clim(vmin, vmax)
		axes_flat[idx + 1].set_title(f"Action {idx}: {action_labels[idx]}, p={current_p[idx]:.4f}")

	cbar.update_normal(im_objects[1])
	fig.canvas.draw_idle()

def on_key(event):
	global current_data, current_index, show_signed
	global current_frame_img, current_maps, current_p

	needs_recompute = False

	if event.key in ("enter", " "):
		pick_random_frame()
		needs_recompute = True
	elif event.key in {"right", "shift+right", "ctrl+right", "left", "shift+left", "ctrl+left"}:
		if   event.key == "right"       : step =  1
		elif event.key == "shift+right" : step = 10
		elif event.key == "ctrl+right"  : step = fps
		elif event.key == "left"        : step = -1
		elif event.key == "shift+left"  : step = -10
		elif event.key == "ctrl+left"   : step = -fps

		current_index = clamp(current_index + step, 0, len(current_data) - 1)
		needs_recompute = True
	elif event.key == "home":
		current_index = 0
		needs_recompute = True
	elif event.key == "end":
		current_index = len(current_data) - 1
		needs_recompute = True
	elif event.key == "v":
		show_signed = not show_signed
		update_display()
		return
	else:
		# ignore all other keys
		return

	if needs_recompute:
		frame_data = current_data[current_index]
		current_frame_img, current_maps, current_p = compute_frame(frame_data)
		update_display()

fig.canvas.mpl_connect("key_press_event", on_key)

update_display()
plt.show()