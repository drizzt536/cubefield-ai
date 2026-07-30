The AI is meant to be played [here](https://cubefield.game-files.crazygames.com/ruffle/cubefield/1/cubefield.html?v=2.6) / [here](https://www.crazygames.com/game/cubefield). Only works on Windows.
The primary monitor should be 2560x1600 with the game on fullscreen or it probably won't work.

prerequisite packages: tensorflow, numpy, opencv-python, dxcam, pytesseract, and pydirectinput.
tensorflow doesn't work on Python 3.14 yet.
Tesseract OC also needs to be installed.

intructions to run player:
1. open the game in fullscreen
2. run `python play.py --autoretry`
3. put focus back on the game window
4. wait until the program is done warming up and press space

press and hold escape whenever done.
