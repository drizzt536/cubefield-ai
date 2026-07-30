from common import np, model_files, load_model
import scipy.ndimage
from sys import argv

model_file = None

for arg in argv[1:]:
	if arg == "--help":
		print(
			"usage: python bin-random.py [--help | FILE | --type=TYPE]"
			"\n"
			"\npass uniform random data into the model and count how often it returns each result."
			"\nFILE is a path to a keras model file. the default is `./game_model.keras`."
			"\nTYPE can be 'solid', 'static', or 'blobby'"
			"\nrequires tensorflow and scipy"
		)

		exit(0)
	elif arg.startswith("--type="):
		IMG_TYPE = arg[len("--type="):]
	else:
		if model_file is not None:
			raise ValueError("multiple model files cannot be passed")

		if not arg:
			raise ValueError("model file cannot be an empty string")

		model_file = arg

model = load_model(model_file or model_files[0])

n = 32_000

match IMG_TYPE:
	case "solid":
		brightness = np.random.uniform(0, 1, (n, 1, 1, 1)).astype(np.float32)

		imgs = np.full((n, 100, 160, 1), brightness, dtype=np.float32)
		t = np.random.uniform(0, 1, (n,)).astype(np.float32)
		del brightness

	case "static":
		imgs = np.random.uniform(0, 1, (n, 100, 160, 1)).astype(np.float32)
		t = np.random.uniform(0, 1, (n,)).astype(np.float32)

	case "blobby":
		# Dependency-free Perlin approximation (Low-frequency spatial noise)
		# We generate tiny 10x16 noise, then smoothly upscale it 10x.
		# This creates connected, cloud-like structural shapes instantly.
		small_noise = np.random.uniform(0, 1, (n, 10, 16, 1))
		imgs = scipy.ndimage.zoom(small_noise, (1, 10, 10, 1), order=1).astype(np.float32)
		t = np.random.uniform(0, 1, (n,)).astype(np.float32)

	case _:
		raise ValueError(f"unrecognized type '{IMG_TYPE}'")

y = model.predict([imgs, t], batch_size=128)

res = np.bincount(y.argmax(axis=1), minlength=3)
res = res / res.sum() # /= doesn't work.

print(f"number of random inputs: {n}")
print(f"average pixel brightness: {imgs.mean()}")
print(f"average time elapsed: {t.mean()}")
print(f"probabilities: none={res[0]:.4f}, left={res[1]:.4f}, right={res[2]:.4f}")
