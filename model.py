from common import Path as _Path, model_files, model_folder
from shutil import copy as file_copy
import filecmp

### helpers start

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

def _restore_named(path: Path, label: str) -> None:
	"restore `path` as the current model, reusing an existing backup slot if it matches one."

	if not path.exists():
		fatal(f"{label} does not exist")

	for n, f in enumerate(model_files):
		f = Path(f)

		if f.exists() and filecmp.cmp(f, path, shallow=False):
			if n != 0:
				print(f"{label} matches backup {n}")

			restore(n)
			return

	# model is not in the chain
	age_chain()
	file_copy(path, model_files[0])
	print(f"restored {label}")

### helpers end

def ls(glob: str = "*.keras.xz") -> None:
	from datetime import datetime
	from hashlib import sha256

	print(f"{"hash":8}  {"modify time":16}  {"file size":11}  file name\n" + "-"*66)
	for path in sorted(Path(model_folder).glob(glob), key=lambda p: p.name):
		stat   = path.stat()
		size   = round(stat.st_size / 1024)
		digest = sha256(path.read_bytes()).hexdigest()[:8]
		mtime  = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

		print(f"{digest}  {mtime}  {f"{size:,} KiB":11}  {path.name}")

def restore(n: int) -> None:
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

def restore_pb() -> None:
	best = find_best_pb()
	_restore_named(best, f"PB model ({best.name})")

def restore_mark(name: str) -> None:
	src = mark_path(name)

	if not src.exists():
		fatal(f"no mark named '{name}'")

	_restore_named(src, f"mark '{name}'")

def swap_paths(a: Path, b: Path) -> None:
	"swap whatever is at `a` and `b`, whether or not either one exists."

	if a == b:
		return

	tmp = a.with_name(a.name + ".tmp")

	a.replace(tmp)
	b.replace(a)
	tmp.replace(b)

def swap(n1: int, n2: int) -> None:
	"swap backups n1 and n2 (0 = current)."

	if n1 == n2:
		print(f"self-swap ({n1} => {n2}) skipped")
		return

	for n in (n1, n2):
		if n >= len(model_files):
			fatal(f"backup {n} doesn't exist. valid range is [0, {len(model_files) - 1}].")

	p1 = Path(model_files[n1])
	p2 = Path(model_files[n2])

	for n, p in ((n1, p1), (n2, p2)):
		if not p.exists():
			fatal(f"backup {n} doesn't exist. skipping swap")

	swap_paths(p1, p2)
	print(f"swapped backups {n1} and {n2}")

def swap_mark(name: str, n: int) -> None:
	"swap backup n (0 = current) with the mark named `name`."

	if n >= len(model_files):
		fatal(f"backup {n} doesn't exist. valid range is [0, {len(model_files) - 1}].")

	p1 = mark_path(name)
	p2 = Path(model_files[n])

	if not p1.exists():
		fatal(f"no mark named '{name}'")

	if not p2.exists():
		fatal(f"backup {n} doesn't exist. skipping swap")

	swap_paths(p1, p2)
	print(f"swapped mark '{name}' and backup {n}")

def swap_mark_mark(name1: str, name2: str) -> None:
	"swap the marks named `name1` and `name2`."

	if name1 == name2:
		print(f"self-swap ('{name1}' => '{name2}') skipped")
		return

	p1 = mark_path(name1)
	p2 = mark_path(name2)

	if not p1.exists():
		fatal(f"no mark named '{name1}'")

	if not p2.exists():
		fatal(f"no mark named '{name2}'")

	swap_paths(p1, p2)
	print(f"swapped marks '{name1}' and '{name2}'")

def mark_set(name: str, n: int = 0) -> None:
	current = Path(model_files[n])

	if not current.exists():
		fatal(f"backup {n} doesn't exist. nothing to mark")

	dst = mark_path(name)

	if dst.exists():
		warn(f"mark '{name}' already exists. overwriting")

	file_copy(current, dst)
	print(f"marked current model as '{name}'")

def mark_mv(old: str, new: str) -> None:
	old_path = mark_path(old)
	new_path = mark_path(new)

	if not old_path.exists():
		fatal(f"no mark named '{old}'")

	if new_path.exists():
		warn(f"mark '{new}' already exists. overwriting")

	old_path.replace(new_path)
	print(f"renamed mark '{old}' to '{new}'")

def mark_cp(old: str, new: str) -> None:
	old_path = mark_path(old)
	new_path = mark_path(new)

	if not old_path.exists():
		fatal(f"no mark named '{old}'")

	if new_path.exists():
		warn(f"mark '{new}' already exists. overwriting")

	file_copy(old_path, new_path)
	print(f"copied mark '{old}' to '{new}'")

def drop_newest(n: int = 1) -> None:
	if n < 0:
		fatal("`n` must be non-negative")

	if n == 0:
		print(f"dropped 0 models")
		return

	existing_count = sum(Path(f).exists() for f in model_files)

	if n >= existing_count:
		fatal(f"can't remove {n} models; only {existing_count} exist")

	# delete the files being dropped
	for i in range(n):
		Path(model_files[i]).unlink()

	# transfer older slots upwards
	for i in range(n, len(model_files)):
		src = Path(model_files[i])
		dst = model_files[i - n]

		if src.exists():
			src.replace(dst)

	print(f"dropped the {n} newest model{'' if n == 1 else 's'}")

def drop_oldest(n: int = 1) -> None:
	existing = [Path(f) for f in model_files if Path(f).exists()]

	if n < 0:
		fatal("`n` must be non-negative")

	if n == 0:
		print(f"dropped 0 models")
		return

	if n >= len(existing):
		fatal(f"can't remove {n} models; only {len(existing)} exist")

	for path in existing[-n:]:
		path.unlink()

	print(f"dropped the {n} oldest model{'' if n == 1 else 's'}")

def rm(n: int) -> None:
	if n == 0:
		drop_newest(1)
		return

	if n < 0 or n >= len(model_files):
		fatal(f"backup {n} doesn't exist. valid range is [0, {len(model_files) - 1}]")

	target = Path(model_files[n])

	if not target.exists():
		fatal(f"'{target}' doesn't exist. nothing to remove")

	existing_count = sum(Path(f).exists() for f in model_files)

	if existing_count <= 1:
		fatal("can't remove the only model in the chain")

	target.unlink()

	for i in range(n, len(model_files) - 1):
		src = Path(model_files[i + 1])

		if src.exists():
			src.replace(model_files[i])

	print(f"removed backup {n}")

def rm_mark(name: str) -> None:
	path = mark_path(name)

	if not path.exists():
		print(f"no mark named '{name}'. nothing to remove")
		return

	path.unlink()
	print(f"removed mark '{name}'")

def clear(glob: str = "*.keras.xz") -> None:
	for path in Path(model_folder).glob(glob):
		# never delete the current model

		if path.name != "game_model.keras.xz":
			path.unlink()

def _str_to_int(n_str: str) -> int:
	try:
		n = int(n_str)

		if n < 0:
			raise ValueError
	except ValueError:
		fatal(f"invalid argument: '{n_str}'. not an integer")

	return n

if __name__ == "__main__":
	from sys import argv

	usage = (
		f"usage: python {Path(argv[0]).name} <command>"
		"\n"
		"\nKeras+xz model local version control program."
		"\nuse to avoid involving git in frequently-changed binaries."
		"\n"
		"\ncommands:"
		"\n    help, --help, -h     print this message and exit"
		"\n"
		# `model restore n` run n times is a noop.
		"\n    ls [all]             list all models. hashes are the first 4 bytes from sha256"
		"\n    ls pb                list PB model(s)"
		"\n    ls backups           list main-line backup models"
		"\n    ls marks             list marked models"
		"\n"
		"\n    restore <n>          restore the model from n backups ago"
		"\n    restore pb           restore the PB model"
		"\n    restore mark <name>  restore a previously marked model"
		"\n"
		"\n    swap <n1> [<n2>]               swap backups <n1> and <n2>. <n2> defaults to 0"
		"\n    swap mark <name> [<n>]         swap mark <name> and backup <n>. <n> defaults to 0"
		"\n    swap <n> mark <name>           swap backup <n> and mark <name>"
		"\n    swap mark <name1> mark <name2> swap marks <name1> and <name2>"
		"\n"
		"\n    mark set <name> [<n>]  mark and save backup n. default is 0"
		"\n    mark mv <old> <new>    remark a model under a different name"
		"\n    mark cp <old> <new>    copy a marked model to a different name"
		"\n"
		"\n    drop [newest] [<n>]  delete the <n> most recent main-line models. default is 1"
		"\n    drop oldest [<n>]    delete the <n> least recent main-line models. default is 1"
		"\n"
		"\n    rm <n>               remove backup <n> and shift up subsequent backups"
		"\n    rm mark <name>       delete a marked model"
		"\n"
		"\n    clear all            delete all models except the current one"
		"\n    clear pb             delete PB models"
		"\n    clear backups        delete main-line backup models"
		"\n    clear marks          delete marked models"
		"\n"
		"\nNOTE: backup 0 is the current model"
	)

	match argv[1:]:
		case [] | ["--help" | "-h" | "help"]:
			print(usage)
			exit(0)

		case ["ls", "all"] | ["ls"] : ls("*.keras.xz")
		case ["ls", "pb"]           : ls("pb-model-*.keras.xz")
		case ["ls", "backups"]      : ls("game_model.old*.keras.xz")
		case ["ls", "marks"]        : ls("model-mark-*.keras.xz")

		case ["restore", "pb"]         : restore_pb()
		case ["restore", "mark", name] : restore_mark(name)
		case ["restore", n_str]        : restore(_str_to_int(n_str))

		case ["swap", "mark", name1, "mark", name2] : swap_mark_mark(name1, name2)
		case ["swap", "mark", name, n_str]          : swap_mark(name, _str_to_int(n_str))
		case ["swap", "mark", name]                 : swap_mark(name, 0)
		case ["swap", n_str, "mark", name]          : swap_mark(name, _str_to_int(n_str))
		case ["swap", n1_str, n2_str]               : swap(_str_to_int(n1_str), _str_to_int(n2_str))
		case ["swap", n_str]                        : swap(_str_to_int(n_str), 0)

		case ["mark", "set", name]        : mark_set(name, 0)
		case ["mark", "set", name, n_str] : mark_set(name, _str_to_int(n_str))
		case ["mark", "mv", old, new]     : mark_mv(old, new)
		case ["mark", "cp", old, new]     : mark_cp(old, new)

		case ["drop", "oldest"]                          : drop_oldest(1)
		case ["drop", "oldest", n_str]                   : drop_oldest(_str_to_int(n_str))
		case ["drop", "newest"]        | ["drop"]        : drop_newest(1)
		case ["drop", "newest", n_str] | ["drop", n_str] : drop_newest(_str_to_int(n_str))

		case ["rm", n_str]        : rm(_str_to_int(n_str))
		case ["rm", "mark", name] : rm_mark(name)

		case ["clear", "all"]      : clear("*.keras.xz")
		case ["clear", "pb"]      : clear("pb-model-*.keras.xz")
		case ["clear", "marks"]   : clear("model-mark-*.keras.xz")
		case ["clear", "backups"] : clear("game_model.old*.keras.xz")

		case _:
			print(usage)
			fatal(f"unrecognized argument list: {' '.join(argv[1:])}")
