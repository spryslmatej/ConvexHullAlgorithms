
POINTS = ../generator/out/points.data
HULL_CPU = ../cpu/out/hull.data
HULL_GPU = ../gpu/out/hull.data

DRAW = drawPoints.py
PDF = pdfPoints.py


# MAX_VAL = long long int max
# DRAW_SCALE = 0.0000001

# MAX_VAL = int max
DRAW_SCALE = 0.0000002

# MAX_VAL = 100
# DRAW_SCALE = 5


drawPoints:
	python3 $(DRAW) $(POINTS) $(DRAW_SCALE)

drawCPU:
	python3 $(DRAW) $(POINTS) $(HULL_CPU) $(DRAW_SCALE)

drawGPU:
	python3 $(DRAW) $(POINTS) $(HULL_GPU) $(DRAW_SCALE)

pointsPDF:
	python3 pdfPoints.py ../generator/out/points.data
