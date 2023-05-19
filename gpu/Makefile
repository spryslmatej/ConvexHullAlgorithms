OUT = out
BIN = bin
OBJ = obj

REQ_FOLDERS = $(OUT) $(OBJ) $(BIN) \
	$(OBJ)/model \
	$(OBJ)/utilities \
	$(OBJ)/quickhull \
	$(OBJ)/crawlers \
	$(OBJ)/concurrent

requiredFolders:
	mkdir -p $(REQ_FOLDERS)

clear:
	rm -rf $(REQ_FOLDERS)

POINTS = ../generator/out/points.data
HULL = $(OUT)/hull.data
MAIN = $(BIN)/main

OBJS = \
	$(OBJ)/main.o \
	$(OBJ)/quickhull/quickhull.o \
	$(OBJ)/quickhull/quickhullWithCrawlers.o \
	$(OBJ)/quickhull/qhData.o \
	$(OBJ)/quickhull/firstSplit.o \
	$(OBJ)/quickhull/recursive.o \
	$(OBJ)/quickhull/helpers.o \
	$(OBJ)/quickhull/cPoint.o \
	$(OBJ)/quickhull/partition.o \
	$(OBJ)/crawlers/crawl.o \
	$(OBJ)/concurrent/concurrent.o \
	$(OBJ)/concurrent/concData.o \
	$(OBJ)/concurrent/crawl.o \
	$(OBJ)/concurrent/grahamScan.o \
	$(OBJ)/concurrent/jarvisMarch.o \
	$(OBJ)/concurrent/partition.o \
	$(OBJ)/model/point.o \
	$(OBJ)/model/directions.o \
	$(OBJ)/model/turn.o \
	$(OBJ)/utilities/helpers.o \
	$(OBJ)/utilities/partition.o

NVCC_FLAGS = -std=c++17 -Xptxas -O3 --extended-lambda

GRID_DIM = 5

# ----------
# COMPILE
# ----------

compile: requiredFolders $(OBJS)
	nvcc $(OBJS) $(NVCC_FLAGS) -o $(MAIN)

$(OBJ)/%.o: src/%.cu
	nvcc -x cu $(NVCC_FLAGS) -dc $< -o $@

# ----------
# QUICKHULL
# ----------

qh: requiredFolders compile runQH

runQH:
	./$(MAIN) -QH $(GRID_DIM) $(POINTS) $(HULL)

# ----------
# QUICKHULL WITH CRAWLERS
# ----------

qhc: requiredFolders compile runQHC

runQHC:
	./$(MAIN) -QHC $(GRID_DIM) $(POINTS) $(HULL)

# ----------
# CONCURRENT HULL
# ----------

co: requiredFolders compile runCO

runCO:
	./$(MAIN) -CO $(GRID_DIM) $(POINTS) $(HULL)
