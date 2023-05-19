from tkinter import *
from matplotlib import pyplot as plt
import numpy as np
import sys

canvas_width = 10
canvas_height = 10
canvas_dot_size = 5

def paint(x, y, color):
    plt.plot(x, y, marker="o", markersize=canvas_dot_size, markeredgecolor=color, markerfacecolor=color)

def readPoints(path, fileType):
    points = []
    import csv
    with open(path,'r') as f:
        reader = csv.reader(f,delimiter=' ')
        if fileType == "input":
            next(reader)
            next(reader)

        for row in reader:
            points.append((row[0], row[1]))
    return points


# === PROGRAM ===

if len(sys.argv) < 2 or len(sys.argv) > 2:
    print("Arguments: <Path to points>")
    exit()

plt.rcParams["figure.figsize"] = [canvas_width, canvas_height]
plt.rcParams["figure.autolayout"] = True
plt.xlim(0, 2147483647)
plt.ylim(0, 2147483647)
plt.axis('off')

plt.tight_layout(pad=0)

# Draw points
points_path = str(sys.argv[1])
points = readPoints(points_path, "input")
for item in points:
    paint(int(item[0]), int(item[1]), "#000000")

# Output
plt.autoscale()
plt.savefig("asdf.pdf", bbox_inches='tight')
