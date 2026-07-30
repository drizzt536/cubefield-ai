from common import Path as _Path, model_files, model_folder, backup_history
from shutil import copy as file_copy
from sys import argv
import filecmp

class Path(type(_Path())):
	def __str__(self):
		return super().__str__().replace('\\', '/')

def fatal(msg: str):
	raise SystemExit(f"\x1b[31m{msg}\x1b[m")

def warn(msg: str) -> None:
	print(f"\x1b[38;2;180;100;0m{msg}\x1b[m")

def find_best_pb() -> Path:
	pb_paths = list(Path(model_folder).glob("pb-model-*.keras.xz"))

	if not pb_paths:
		fatal("no PB model found")

	if len(pb_paths) > 1:
		warn(f"more than one PB model found. using '{pb_paths[0]}'")

	return pb_paths[0]

def mark_path(name: str) -> Path:
	if '/' in name or '\\' in name:
		raise ValueError("path separators are not allowed in mark names")

	return Path(model_folder) / f"model-mark-{name}.keras.xz"

def age_chain() -> None:
	"shift every main-line backup back by one slot, freeing up slot 0."

	for i in range(len(model_files) - 1, 0, -1):
		src = Path(model_files[i - 1])

		if src.exists():
			src.replace(model_files[i])

def rotate_models(n: int) -> None:
	"move model_files[n] to the front, shifting 0..n-1 back by one slot."

	if n == 0:
		print("requested model is current")
		return

	if n >= len(model_files):
		fatal(f"backup {n} doesn't exist. valid range is [0, {len(model_files) - 1}].")

	target = Path(model_files[n])

	if not target.exists():
		fatal(f"'{target}' doesn't exist. skipping restore")

	tmp = target.with_name(target.name + ".tmp")
	target.replace(tmp)

	for i in range(n - 1, -1, -1):
		src = Path(model_files[i])

		if src.exists():
			src.replace(model_files[i + 1])

	tmp.replace(model_files[0])
	print(f"restored model from backup {n}")

def restore_model(path: Path, label: str) -> None:
	"restore `path` as the current model, reusing an existing backup slot if it matches one."

	for n, f in enumerate(model_files):
		f = Path(f)

		if f.exists() and filecmp.cmp(f, path, shallow=False):
			if n != 0:
				print(f"{label} matches backup {n}")

			rotate_models(n)
			return

	# model is not in the chain
	age_chain()
	file_copy(path, model_files[0])
	print(f"restored {label}")

def restore_pb() -> None:
	best = find_best_pb()
	restore_model(best, f"PB model ({best.name})")

def restore_mark(name: str) -> None:
	src = mark_path(name)

	if not src.exists():
		fatal(f"no mark named '{name}'")

	restore_model(src, f"mark '{name}'")

def mark_set(name: str) -> None:
	current = Path(model_files[0])

	if not current.exists():
		fatal(f"'{current}' doesn't exist. no model to mark")

	dst = mark_path(name)

	if dst.exists():
		warn(f"mark '{name}' already exists. overwriting")

	file_copy(current, dst)
	print(f"marked current model as '{name}'")

def mark_remove(name: str) -> None:
	path = mark_path(name)

	if not path.exists():
		print(f"no mark named '{name}'. nothing to remove")
		return

	path.unlink()
	print(f"removed mark '{name}'")

def list_models(glob: str = "*.keras.xz") -> None:
	for path in Path(model_folder).glob(glob):
		print(path)

def clear_models(glob: str = "*.keras.xz") -> None:
	for path in Path(model_folder).glob(glob):
		# never delete the current model

		if path.name != "game_model.keras.xz":
			path.unlink()

def drop_models(n: int = 1) -> None:
	if n < 0:
		fatal("`n` must be non-negative")

	if n == 0:
		print(f"dropped 0 models")
		return

	existing_count = sum(Path(f).exists() for f in model_files)
	max_droppable = existing_count - 1

	if max_droppable <= 0:
		fatal("can't drop the only model in the chain")

	n = min(n, max_droppable)

	# delete the file being dropped
	for i in range(n):
		current = Path(model_files[i])
		if current.exists():
			current.unlink()

	# transfer older slots upwards
	for i in range(n, len(model_files)):
		src = Path(model_files[i])
		dst = model_files[i - n]

		if src.exists():
			src.replace(dst)

	print(f"dropped {n} model{'' if n == 1 else 's'}")

if __name__ == "__main__":
	usage = (
		f"usage: python {Path(argv[0]).name} <command>"
		"\n"
		"\nmodel local version control program."
		"\nuse to avoid involving git in frequently-changed binaries."
		"\n"
		"\ncommands:"
		"\n    help | --help | -h   print this message and exit"
		"\n"
		"\n    restore backup <n>   restore the model from n backups ago (0 = current)"
		"\n    restore pb           restore the PB model"
		"\n    restore mark <name>  restore a previously marked model"
		"\n"
		"\n    drop [<n>]           drop the <n> most recent main-line models. default is 1"
		"\n"
		"\n    ls [all]             list all models"
		"\n    ls pb                list PB models"
		"\n    ls backups           list main-line backup models"
		"\n    ls marks             list marked models"
		"\n"
		"\n    clear all            delete all models except the current one"
		"\n    clear pb             delete PB models"
		"\n    clear backups        delete main-line backup models"
		"\n    clear marks          delete marked models"
		"\n"
		"\n    mark set <name>      save the current model under a name for later restoring"
		"\n    mark rm <name>       delete a marked model (does not affect the current model)"
	)

	match argv[1:]:
		case [] | ["--help" | "-h" | "help"]:
			print(usage)
			exit(0)

		case ["restore", "pb"]: restore_pb()
		case ["restore", "mark", name]: restore_mark(name)
		case ["restore", n_str]:
			try:
				n = int(n_str)

				if n < 0:
					raise ValueError
			except ValueError:
				fatal(f"invalid argument: '{n_str}'")

			rotate_models(n)

		case ["drop"]:
			drop_models(1)
		case ["drop", n_str]:
			try:
				n = int(n_str)

				if n < 0:
					raise ValueError
			except ValueError:
				fatal(f"invalid argument: '{n_str}'")

			drop_models(n)

		case ["ls", "all"] | ["ls"] : list_models("*.keras.xz")
		case ["ls", "pb"]      : list_models("pb-model-*.keras.xz")
		case ["ls", "backups"] : list_models("game_model.old*.keras.xz")
		case ["ls", "marks"]   : list_models("model-mark-*.keras.xz")

		case ["list", "all"]      : clear_models("*.keras.xz")
		case ["clear", "pb"]      : clear_models("pb-model-*.keras.xz")
		case ["clear", "marks"]   : clear_models("model-mark-*.keras.xz")
		case ["clear", "backups"] : clear_models("game_model.old*.keras.xz")

		case ["mark", "set", name]: mark_set(name)
		case ["mark", "rm", name]: mark_remove(name)

		case _:
			print(usage)
			fatal(f"unrecognized argument list: {' '.join(argv[1:])}")
