# this is probably best to do with GPU enabled (e.g. in WSL2)

from common import model_files, np, os, Path, get_paths, count_usages, save_model
from model import age_chain as age_model_chain
import tensorflow as tf
keras = tf.keras
layers = keras.layers

usages = count_usages(p_intrin=0.9)
paths  = list(get_paths())

np.random.default_rng().shuffle(paths)

sizes = [p.stat().st_size for p in paths]
total = sum(sizes)
target_val = 0.1 * total

val_paths, running = [], 0
for p, size in zip(paths, sizes):
	if running < target_val:
		val_paths.append(p)
		running += size
	else:
		break

val_set = set(val_paths)
train_paths = [p for p in paths if p not in val_set]

print(
	f"/// {len(train_paths)} train files, "
	f"{len(val_paths)} val files "
	f"({running/1e6:.1f} MB, {running/total:.1%} of total)"
)

def make_dataset(paths, shuffle_buffer=None, batch=32):
	def gen():
		for path in paths:
			run = np.load(path)

			elapsed = run["elapsed"]
			labels  = np.zeros(len(elapsed), dtype=np.int8)

			labels[run["left"]]  = 1
			labels[run["right"]] = 2

			mask = (labels != 0) | (np.random.rand(len(labels)) > usages["drop"])

			frames  = run["frames"][mask][..., None]
			labels  = labels[mask]
			elapsed = (elapsed[mask].astype(np.float32) / 300)[..., None]

			yield {"frame": frames, "elapsed": elapsed}, labels

	output_signature = (
		{
			"frame":   tf.TensorSpec(shape=(None, 100, 160, 1), dtype=tf.uint8),
			"elapsed": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
		},
		tf.TensorSpec(shape=(None,), dtype=tf.int8),
	)

	ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
	ds = ds.unbatch()

	if shuffle_buffer:
		ds = ds.shuffle(shuffle_buffer)

	ds = ds.batch(batch)
	ds = ds.map(lambda x, y: ({**x, "frame": tf.cast(x["frame"], tf.float32) / 255}, y))
	ds = ds.prefetch(tf.data.AUTOTUNE)
	return ds

BATCH  = 32
BUFFER = 50_000

train_ds = make_dataset(train_paths, shuffle_buffer=BUFFER, batch=BATCH)
val_ds   = make_dataset(val_paths, batch=BATCH)

frame_input   = keras.Input(shape=(100, 160, 1), name="frame")
elapsed_input = keras.Input(shape=(1,), name="elapsed")

# convolution stack
conv_stack = keras.Sequential([
	layers.Conv2D(32, kernel_size=3, strides=1, padding="same", activation="relu"),
	layers.AveragePooling2D(pool_size=2),

	layers.Conv2D(32, kernel_size=3, strides=1, padding="same", activation="relu"),
	layers.AveragePooling2D(pool_size=2),

	layers.Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu"),
	layers.AveragePooling2D(pool_size=2),

	layers.Flatten(),
	layers.Dropout(0.10),
], name="conv_stack")

t = layers.Dense(4)(elapsed_input)

x = conv_stack(frame_input)
x = layers.Concatenate()([x, t])
x = layers.Dense(8, activation="relu")(x)
output = layers.Dense(3, activation="softmax", name="outputs")(x)

model = keras.Model(inputs=[frame_input, elapsed_input], outputs=output)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

early_stop = keras.callbacks.EarlyStopping(
	monitor="val_loss",
	patience=3, # stop if val_loss doesn't improve for 3 epochs in a row
	restore_best_weights=True,
)

model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=20,
	# steps_per_epoch=usages["total"] // BATCH,
	shuffle=False, # dataset already shuffles via .shuffle()
	callbacks=[early_stop]
)

age_model_chain()
save_model(model, model_files[0])
