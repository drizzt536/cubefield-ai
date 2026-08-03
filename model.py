from common import Path as _Path, model_files, model_folder
from shutil import copy as file_copy

### helpers start

class Path(type(_Path())):
	def __str__(self):
		return super().__str__().replace('\\', '/')

def fatal(msg: str):
	raise SystemExit(f"\x1b[31m{msg}\x1b[m")

def warn(msg: str) -> None:
	print(f"\x1b[38;2;180;100;0m{msg}\x1b[m")

def pb_path() -> Path:
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

		if f.exists() and cmp_paths(f, path, silent=True):
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

def cmp_paths(path1: Path | str, path2: Path | str, *, silent: bool = False) -> bool:
	import filecmp

	same = filecmp.cmp(path1, path2, shallow=False)

	if not silent:
		print(f"models are {"identical" if same else "different"}")

	return same

def cmp(n1: int, n2: int, *, silent: bool = False) -> bool:
	"compare backups n1 and n2"

	if n1 == n2:
		print("models are identical")
		return

	for n in (n1, n2):
		if n >= len(model_files):
			fatal(f"backup {n} doesn't exist. valid range is [0, {len(model_files) - 1}].")

	p1 = Path(model_files[n1])
	p2 = Path(model_files[n2])

	for n, p in ((n1, p1), (n2, p2)):
		if not p.exists():
			fatal(f"backup {n} doesn't exist. skipping compare")

	return cmp_paths(p1, p2, silent=silent)

def cmp_pb(n: int, *, silent: bool = False) -> bool:
	"compare backup n with the PB model"

	if n >= len(model_files):
		fatal(f"backup {n} doesn't exist. valid range is [0, {len(model_files) - 1}].")

	p1 = pb_path()
	p2 = Path(model_files[n])

	if not p2.exists():
		fatal(f"backup {n} doesn't exist. skipping compare")

	return cmp_paths(p1, p2, silent=silent)

def cmp_mark(name: str, n: int, *, silent: bool = False) -> bool:
	"compare backup n with mark `name`"

	if n >= len(model_files):
		fatal(f"backup {n} doesn't exist. valid range is [0, {len(model_files) - 1}].")

	p1 = mark_path(name)
	p2 = Path(model_files[n])

	if not p1.exists():
		fatal(f"no mark named '{name}'. skipping compare")

	if not p2.exists():
		fatal(f"backup {n} doesn't exist. skipping compare")

	return cmp_paths(p1, p2, silent=silent)

def cmp_pb_mark(name: str, *, silent: bool = False) -> bool:
	"compare backup n with mark `name`"

	p1 = pb_path()
	p2 = mark_path(name)

	if not p2.exists():
		fatal(f"no mark named '{name}'. skipping compare")

	return cmp_paths(p1, p2, silent=silent)

def cmp_mark_mark(name1: str, name2: str, *, silent: bool = False) -> bool:
	"compare marks `name1` and `name2`"

	if name1 == name2:
		print("models are identical")
		return

	p1 = mark_path(name1)
	p2 = mark_path(name2)

	if not p1.exists():
		fatal(f"no mark named '{name1}'. skipping compare")

	if not p2.exists():
		fatal(f"no mark named '{name2}'. skipping compare")

	return cmp_paths(p1, p2, silent=silent)

def swap_paths(a: Path, b: Path) -> None:
	"unconditional swap. assumes both paths exist"

	if a == b:
		return

	tmp = a.with_name(a.name + ".tmp")

	a.replace(tmp)
	b.replace(a)
	tmp.replace(b)

def swap(n1: int, n2: int) -> None:
	"swap backups n1 and n2"

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
	"swap backup n with mark `name`"

	if n >= len(model_files):
		fatal(f"backup {n} doesn't exist. valid range is [0, {len(model_files) - 1}].")

	p1 = mark_path(name)
	p2 = Path(model_files[n])

	if not p1.exists():
		fatal(f"no mark named '{name}'. skipping swap")

	if not p2.exists():
		fatal(f"backup {n} doesn't exist. skipping swap")

	swap_paths(p1, p2)
	print(f"swapped mark '{name}' and backup {n}")

def swap_mark_mark(name1: str, name2: str) -> None:
	"swap marks `name1` and `name2`"

	if name1 == name2:
		print(f"self-swap ('{name1}' => '{name2}') skipped")
		return

	p1 = mark_path(name1)
	p2 = mark_path(name2)

	if not p1.exists():
		fatal(f"no mark named '{name1}'. skipping swap")

	if not p2.exists():
		fatal(f"no mark named '{name2}'. skipping swap")

	swap_paths(p1, p2)
	print(f"swapped marks '{name1}' and '{name2}'")

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
	best = pb_path()
	_restore_named(best, f"PB model ({best.name})")

def restore_mark(name: str) -> None:
	src = mark_path(name)

	if not src.exists():
		fatal(f"no mark named '{name}'")

	_restore_named(src, f"mark '{name}'")

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

def _str2int(n_str: str) -> int:
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
		"\n    help, --help, -h       print this message and exit"
		"\n"
		# `model restore n` run n times is a noop.
		"\n    ls [all]               list all models. hashes are the first 4 bytes from sha256"
		"\n    ls pb[s]               list PB model(s)"
		"\n    ls current             list the current model file"
		"\n    ls main[line]          list all main-line history models"
		"\n    ls backup[s]           list main-line backup models"
		"\n    ls mark[s]             list marked models"
		"\n"
		"\n    cmp <A> [<B>]          compare model files. <A> and <B> can each be `<n>` (backup <n>),"
		"\n                           or `mark <name>`. <B> defaults to 0"
		"\n"
		"\n    swap <A> [<B>]         swap model files. <A> and <B> can each be `<n>` (backup <n>),"
		"\n                           or `mark <name>`. <B> defaults to 0"
		"\n"
		"\n    restore <n>            restore the model from <n> backups ago"
		"\n    restore pb             restore the PB model"
		"\n    restore mark <name>    restore a previously marked model"
		"\n"
		"\n    mark set <name> [<n>]  mark and save backup <n>. <n> defaults to 0"
		"\n    mark mv <old> <new>    remark a model under a different name"
		"\n    mark cp <old> <new>    copy a marked model to a different name"
		"\n"
		"\n    drop [newest] [<n>]    delete the <n> most recent main-line models. <n> defaults to 1"
		"\n    drop oldest [<n>]      delete the <n> least recent main-line models. <n> defaults to 1"
		"\n"
		"\n    rm <n>                 remove backup <n> and shift up subsequent backups"
		"\n    rm mark <name>         delete a marked model"
		"\n"
		"\n    clear all              delete all models except the current one"
		"\n    clear pb               delete PB model(s)"
		"\n    clear backups          delete main-line backup models"
		"\n    clear marks            delete marked models"
		"\n"
		"\nNOTE: backup 0 is the current model"
	)

	match argv[1:]:
		case [] | ["--help" | "-h" | "help"]:
			print(usage)
			exit(0)

		# n => int, N (name) => string

		case ["ls", "all"] | ["ls"]       : ls("*.keras.xz")
		case ["ls", "current"]            : ls("game_model.keras.xz")
		case ["ls", "pbs" | "pb"]         : ls("pb-model-*.keras.xz")
		case ["ls", "backups" | "backup"] : ls("game_model.old*.keras.xz")
		case ["ls", "mainline" | "main"]  : ls("game_model*.keras.xz")
		case ["ls", "marks" | "mark"]     : ls("model-mark-*.keras.xz")

		case ["cmp", "mark", N1, "mark", N2] : cmp_mark_mark(N1, N2)
		case ["cmp", "pb", "mark", N]        : cmp_pb_mark(N)
		case ["cmp", "mark", N, "pb"]        : cmp_pb_mark(N)
		case ["cmp", "mark", N, n]           : cmp_mark(N, _str2int(n))
		case ["cmp", n, "mark", N]           : cmp_mark(N, _str2int(n))
		case ["cmp", "mark", N]              : cmp_mark(N, 0)
		case ["cmp", "pb", "pb"]             : print("models are identical")
		case ["cmp", "pb", n]                : cmp_pb(_str2int(n))
		case ["cmp", n, "pb"]                : cmp_pb(_str2int(n))
		case ["cmp", "pb"]                   : cmp_pb(0)
		case ["cmp", n1, n2]                 : cmp(_str2int(n1), _str2int(n2))
		case ["cmp", n]                      : cmp(_str2int(n), 0)

		case ["swap", "mark", N1, "mark", N2] : swap_mark_mark(N1, N2)
		case ["swap", "mark", N, n]           : swap_mark(N, _str2int(n))
		case ["swap", "mark", N]              : swap_mark(N, 0)
		case ["swap", n, "mark", N]           : swap_mark(N, _str2int(n))
		case ["swap", n1, n2]                 : swap(_str2int(n1), _str2int(n2))
		case ["swap", n]                      : swap(_str2int(n), 0)

		case ["restore", "pb"]      : restore_pb()
		case ["restore", "mark", N] : restore_mark(N)
		case ["restore", n]         : restore(_str2int(n))

		case ["mark", "set", N]       : mark_set(N, 0)
		case ["mark", "set", N, n]    : mark_set(N, _str2int(n))
		case ["mark", "mv", old, new] : mark_mv(old, new)
		case ["mark", "cp", old, new] : mark_cp(old, new)

		case ["drop", "oldest"]                  : drop_oldest(1)
		case ["drop", "oldest", n]               : drop_oldest(_str2int(n))
		case ["drop", "newest"]    | ["drop"]    : drop_newest(1)
		case ["drop", "newest", n] | ["drop", n] : drop_newest(_str2int(n))

		case ["rm", n]         : rm(_str2int(n))
		case ["rm", "mark", N] : rm_mark(N)

		case ["clear", "all"]     : clear("*.keras.xz")
		case ["clear", "pb"]      : clear("pb-model-*.keras.xz")
		case ["clear", "marks"]   : clear("model-mark-*.keras.xz")
		case ["clear", "backups"] : clear("game_model.old*.keras.xz")

		case _:
			print(usage)
			fatal(f"unrecognized argument list: {' '.join(argv[1:])}")
