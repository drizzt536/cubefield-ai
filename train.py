# this is probably best to do with GPU enabled (e.g. in WSL2)
# wsl bash -ic "python train.py"

from common import model_files, np, os, Path, get_paths, count_usages, save_model
from model import age_chain as age_model_chain
import tensorflow as tf
keras = tf.keras
layers = keras.layers

BATCH  = 32
BUFFER = 50_000

turn_bias = -0.80
usages = count_usages(turn_bias=turn_bias)
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
	f"/// {len(train_paths)} train files,"
	f" {len(val_paths)} val files"
	f" ({running/1e6:.1f} MB, {running/total:.1%} of total)"
	"\n"
	f"/// datapoints: total={usages["total"]:,},"
	f" none=~{round(usages["none"]*(1 - usages["drop"])):,}/{usages["none"]:,},"
	f" left={usages["left"]:,},"
	f" right={usages["right"]:,},"
	f" drop={usages["drop"]:.4f},"
	f" turn bias={turn_bias}"
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
			"frame"   : tf.TensorSpec(shape=(None, 100, 160, 1), dtype=tf.uint8),
			"elapsed" : tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
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
	layers.Dropout(0.1),
], name="conv_stack")

x = conv_stack(frame_input)
x = layers.Concatenate()([x, elapsed_input])
x = layers.Dense(8, activation="relu")(x)
output = layers.Dense(3, activation="softmax", name="outputs")(x)

model = keras.Model(inputs=[frame_input, elapsed_input], outputs=output)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

early_stop = keras.callbacks.EarlyStopping(
	monitor="val_loss",
	patience=5, # stop if val_loss doesn't improve for <patience> epochs in a row
	restore_best_weights=True,
)

class PerClassAccuracy(keras.callbacks.Callback):
	def __init__(self, num_classes, val_ds):
		super().__init__()
		self.val_ds = val_ds
		self.num_classes = num_classes

		self.confusion_matrices = []
		self.class_accuracies = []

	def on_epoch_end(self, epoch, logs=None):
		if logs is None:
			logs = {}

		y_true = []
		y_pred = []

		for x_batch, y_batch in self.val_ds:
			pred = self.model(x_batch, training=False)
			pred = tf.argmax(pred, axis=1)

			y_true.append(y_batch)
			y_pred.append(pred)

		y_true = tf.concat(y_true, axis=0)
		y_pred = tf.concat(y_pred, axis=0)

		cm = tf.math.confusion_matrix(
			y_true,
			y_pred,
			num_classes=self.num_classes
		).numpy()

		class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

		self.confusion_matrices.append(cm)
		self.class_accuracies.append(class_acc)

		for i, acc in enumerate(class_acc):
			# class accuracy
			logs[f"accuracy{i}"] = float(acc)

class_accuracies = PerClassAccuracy(3, val_ds)

history = model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=100,
	# steps_per_epoch=usages["total"] // BATCH,
	shuffle=False, # dataset already shuffles via .shuffle()
	callbacks=(class_accuracies, early_stop),
	class_weight={
		0: 1.0,
		1: max(1.0, usages["right"] / usages["left"]),
		2: max(1.0, usages["left"] / usages["right"]),
	}
)

def diagnose_training_state_v2(history_dict, confusion_matrices, window=5):
	epochs = len(history_dict["loss"])

	if epochs < 3:
		# 3+ epochs are needed for a meaningful slope
		return "TOO EARLY: Not enough epochs to calculate a trendline."

	window = min(window, epochs)

	# global metrics
	val_loss = np.array(history_dict["val_loss"])
	best_val_idx = np.argmin(val_loss)
	best_val = val_loss[best_val_idx]
	final_val = val_loss[-1]

	val_slope   = np.polyfit(np.arange(window), val_loss[-window:], 1)[0]
	train_slope = np.polyfit(np.arange(window), history_dict["loss"][-window:], 1)[0]

	# per-class trend metrics
	acc1_slope = np.polyfit(np.arange(window), history_dict["accuracy1"][-window:], 1)[0]
	acc2_slope = np.polyfit(np.arange(window), history_dict["accuracy2"][-window:], 1)[0]

	cm = confusion_matrices[best_val_idx]
	# NOTE: indexing is `cm[actual, predicted]`

	print(str(cm)[2:-2].replace("]\n [", '\n'))

	# prevent divide-by-zero if a class is entirely missing in the validation batch
	true_action_count = max(1, np.sum(cm[1]) + np.sum(cm[2]))

	# lazy metric: actual action (1 or 2) predicted as none (0)
	lazy_predictions = cm[1, 0] + cm[2, 0]
	lazy_rate = lazy_predictions / true_action_count

	# reversal metric: left/right predicted as the opposite
	reversal_predictions = cm[1, 2] + cm[2, 1]
	reversal_rate = reversal_predictions / true_action_count

	diagnosis = []

	# check for class collapse (model gives up on steering)
	if history_dict["accuracy1"][-1] < 0.20 or history_dict["accuracy2"][-1] < 0.20:
		diagnosis.append("CRITICAL CLASS COLLAPSE: The model has almost entirely forgotten how to turn one of the directions.")
	elif acc1_slope < -0.05 or acc2_slope < -0.05:
		diagnosis.append("WARNING: Steering accuracy is actively bleeding off. A specific action class is degrading rapidly.")

	# check for behavioral archetypes via confusion matrix
	if lazy_rate > 0.40:
		diagnosis.append(f"BEHAVIOR: LAZY. The model is predicting 'none' on {lazy_rate*100:.1f}% of frames that require steering. (Increase class weights for 1 and 2).")

	if reversal_rate > 0.15:
		diagnosis.append(f"BEHAVIOR: CONFUSED. The model steered the completely wrong direction on {reversal_rate*100:.1f}% of action frames. (Visual features are blurring together).")

	# check for global overfitting / convergence
	threshold = best_val*0.005

	if final_val > best_val*1.05 and val_slope > threshold and train_slope < 0:
		diagnosis.append(f"STATE: OVERFIT. Peak global performance was at epoch {best_val_idx + 1}. Validation loss is rising.")
	elif train_slope < -threshold and val_slope < -threshold:
		diagnosis.append("STATE: STILL LEARNING. Global loss is dropping beautifully.")
	elif abs(train_slope) <= threshold and abs(val_slope) <= threshold:
		diagnosis.append("STATE: CONVERGED. The model has mathematically flatlined and settled into its final state.")
	else:
		diagnosis.append("STATE: FLUCTUATING. The global loss curve is a bit chaotic right now.")

	return "\n".join(diagnosis)

print(diagnose_training_state_v2(
	history.history,
	class_accuracies.confusion_matrices,
	window=early_stop.patience
))

age_model_chain()
save_model(model, model_files[0])
