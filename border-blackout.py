import tkinter as tk
from sys import argv, platform

if platform == "win32":
	import ctypes

	ctypes.windll.user32.SetProcessDPIAware()

if len(argv) < 5:
	print("usage: script.py <top> <right> <bottom> <left>")
	exit(1)

u, r, d, l = map(int, argv[1:5])

root = tk.Tk()
root.withdraw() # hide the unused base window

root.update_idletasks()
w = root.winfo_screenwidth()
h = root.winfo_screenheight()

def make_bar(w: int, h: int, x: int, y: int):
	win = tk.Toplevel(root)
	win.overrideredirect(True)
	win.configure(bg="black")
	win.attributes("-topmost", True)
	win.geometry(f"{w}x{h}+{x}+{y}")

	return win

bars = []
if u > 0: bars.append(make_bar( w , u , 0     , 0     ))
if r > 0: bars.append(make_bar( r , h , w - r , 0     ))
if d > 0: bars.append(make_bar( w , d , 0     , h - d ))
if l > 0: bars.append(make_bar( l , h , 0     , 0     ))

root.bind_all("<Escape>", lambda event: root.destroy())
root.mainloop()
