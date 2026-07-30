from sys import argv

if len(argv) > 1 and argv[1] == "--help":
	print(
		"usage: python time-sweep.py [--help | --random]"
		"\n"
		"\nplots the model outputs over time assuming a fixed image."
		"\nplot happens for t in [-5, 5] with a dt of 0.001."
		"\n"
		"\nthe default is to sweep using an all-black image."
		"\n--random makes it sweep using a random image."
		"\n"
		"\nrequires tensorflow, numpy, and matplotlib"
	)

	exit(0)

import matplotlib.pyplot as plt
import tensorflow as tf
from common import np, model_files, load_model

random = len(argv) > 1 and argv[1] == "--random"

model = load_model(model_files[0])

dt = np.float32(0.001)
t0 = np.float32(-5)
t1 = np.float32(5)
img_shape = (1, 100, 160, 1)

if random:
	img = np.random.uniform(0, 1, img_shape).astype(np.float32)
else:
	img = np.zeros(img_shape, dtype=np.float32)

t    = np.arange(t0, t1, dt, dtype=np.float32)
imgs = np.repeat(img, len(t), axis=0)
y    = model.predict([imgs, t], batch_size=128).T

dydt = np.gradient(y, t, axis=1)

print("generating plots", end="", flush=True)
plt.style.use("dark_background")

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(t, y[0], label="no action")
ax1.plot(t, y[1], label="left")
ax1.plot(t, y[2], label="right")

ax1.axvline(x=0, color="white", linestyle=':')
ax1.axvline(x=1, color="white", linestyle=':')

ax1.set_xlabel("elapsed time/300")
ax1.set_ylabel("model")
ax1.set_title("Model Values")
ax1.legend()

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(t, dydt[0], label="no action")
ax2.plot(t, dydt[1], label="left")
ax2.plot(t, dydt[2], label="right")

ax2.axvline(x=0, color="white", linestyle=':')
ax2.axvline(x=1, color="white", linestyle=':')

ax2.set_xlabel("elapsed time/300")
ax2.set_ylabel("d/dt model")
ax2.set_title("Numerical Derivatives")
ax2.legend()

print("\ropening plots\x1b[K")
plt.show()
