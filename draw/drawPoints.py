from tkinter import *
import numpy as np
import sys

canvas_width = 700
canvas_height = 700
canvas_dot_size = 5

def paint(x, y, color):
    x1, y1 = (
        x * canvas_scale - canvas_dot_size / 2,
        y * canvas_scale - canvas_dot_size / 2,
    )
    x2, y2 = x1 + canvas_dot_size, y1 + canvas_dot_size
    w.create_oval(x1, y1, x2, y2, fill=color)


def lineHull(x, hull, color):
    for item in hull:
        w.create_line(
            int(x[0]) * canvas_scale,
            int(x[1]) * canvas_scale,
            int(item[0]) * canvas_scale,
            int(item[1]) * canvas_scale,
            fill=color,
            width=0.25,
            dash=(6, 10),
        )

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

if len(sys.argv) < 3 or len(sys.argv) > 4:
    print("Arguments: \t<Path to points> <Path to hull> <Draw scale>")
    print("OR: \t\t<Path to points> <Draw scale>")
    exit()

master = Tk()
master.title("Points")
w = Canvas(master, width=canvas_width, height=canvas_height)
w.pack(expand=YES, fill=BOTH)

if len(sys.argv) == 4:
    hull_path = str(sys.argv[2])
    canvas_scale = float(sys.argv[3])
else:
    canvas_scale = float(sys.argv[2])

# Draw points
points_path = str(sys.argv[1])
points = readPoints(points_path, "input")
for item in points:
    paint(int(item[0]), int(item[1]), "#000000")

# Draw hull
if len(sys.argv) == 4:
    hull = readPoints(hull_path, "hull")
    for item in hull:
        paint(int(item[0]), int(item[1]), "#FF0000")

# for item in hull:
#     lineHull(item, hull, "#FF0000")

# for item in points:
#     lineHull(item, hull, "#000000")

mainloop()
