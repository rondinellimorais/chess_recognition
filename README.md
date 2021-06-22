# Link
	- https://jsfiddle.net/q76uzxwe/1/
	- https://www.freecodecamp.org/news/simple-chess-ai-step-by-step-1d55a9266977/

# chess_recognition

Before cloning this repo it depends on the installed `git lfs`

```
brew install git-lfs
```

# Get started

Conda environment

```bash
conda activate chess_recognition
```

## Mapping

```bash
python3 src/main.py --mapping
```

## Start

```bash
python3 src/main.py --mapping
```

# Troubleshooting

## Could not load the Qt platform plugin "xcb"

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, xcb.

Aborted (core dumped)
```

**Resolution**

Just remove the damn

```
~/miniconda3/envs/envname/lib/python-3.9/site-packages/cv2/qt/plugins
```

https://github.com/wkentaro/labelme/issues/842#issuecomment-826481652