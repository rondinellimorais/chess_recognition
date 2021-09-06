# Chess piece Recognition

A computer vision project for playing chess.

<p><strong>Object Detection</strong></p>
<img style="max-width: 60%" src="assets/img/hardware+yolo.jpeg">

<p>
  <br/>
  <strong>Board mapping</strong>
</p>
<img src="assets/img/board_mapping.png">

<p>
  <br/>
  <strong>Find board coordinates</strong>
</p>
<img src="assets/img/find_coordinates.png">

<p>
  <br/>
  <strong>Mapped board</strong>
</p>
<img src="assets/img/mapped_board.jpeg">

# Get started

Before cloning this repo it depends on the installed `git lfs`

```
brew install git-lfs
```

Create conda environment

```bash
# macos
conda env create -f chess_recognition_macos.yml 

# linux
conda env create -f chess_recognition_linux.yml 

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

## Notebooks

| Title     | Description | Link |
|:----------|:-----------|-----:|
| ♟️ Chess Piece Detection | Use [**Darknet**](https://github.com/AlexeyAB/darknet) to detect pieces | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nrTyy-m-xG6vmG6klsLm1dTlLJYQTnrM) |
| 📷 Game Board Mapping | Find the playable area of the board | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13r2HiJeB9G4eQP5a9WTQE_NyeiGQHnF6) |
| 🖼️ Image Data Augmentation | Data augmentation techniques | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NXRUnvztSCs7IljH8vXqEIIsFF0IPaTH) |

# Running

[![Youtube](http://img.youtube.com/vi/9dsYuFIf6_c/0.jpg)](https://www.youtube.com/watch?v=9dsYuFIf6_c "Example 1")

[![Youtube](http://img.youtube.com/vi/3o1dMs6xAT0/0.jpg)](https://www.youtube.com/watch?v=3o1dMs6xAT0 "Example 2")

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

# References
   - https://jsfiddle.net/q76uzxwe/1/
   - https://www.freecodecamp.org/news/simple-chess-ai-step-by-step-1d55a9266977/
