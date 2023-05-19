OUT = out
BIN = bin
OBJ = obj

REQ_FOLDERS = $(OUT) $(OBJ) $(BIN) \
	$(OBJ)/model \
	$(OBJ)/quickhull \
	$(OBJ)/concurrent \
	$(OBJ)/graham \
	$(OBJ)/crawlers \
	$(OBJ)/chan \
	$(OBJ)/jarvis \
	$(OBJ)/utilities \

requiredFolders:
	mkdir -p $(REQ_FOLDERS)

clear:
	rm -rf $(REQ_FOLDERS)

POINTS = ../generator/out/points.data

MAIN = $(BIN)/main
HULL = $(OUT)/hull.data

OBJS= \
	$(OBJ)/main.o \
	$(OBJ)/model/point.o \
	$(OBJ)/model/segment.o \
	$(OBJ)/quickhull/quickhull.o \
	$(OBJ)/quickhull/quickhullWithCrawlers.o \
	$(OBJ)/quickhull/helpers.o \
	$(OBJ)/concurrent/concurrent.o \
	$(OBJ)/crawlers/crawlerSegment.o \
	$(OBJ)/crawlers/crawler.o \
	$(OBJ)/crawlers/crawlers.o \
	$(OBJ)/graham/graham.o \
	$(OBJ)/chan/chan.o \
	$(OBJ)/jarvis/jarvis.o \
	$(OBJ)/utilities/values.o \
	$(OBJ)/utilities/pointPosition.o

GPP_FLAGS = -Wall -pedantic -fopenmp -O3 -std=c++2a

GRID_DIM = 30
THREADS = 4

# ----------
# COMPILE
# ----------

$(OBJ)/%.o: src/%.cpp
	g++ $(GPP_FLAGS) -c $< -o $@
	
compile: requiredFolders $(OBJS)
	g++ $(GPP_FLAGS) $(OBJS) -o $(MAIN)

generate:
	$(MAKE) --directory="../generator" generate

# ----------
# JARVIS
# ----------

j: requiredFolders compile runJ

runJ:
	./$(MAIN) -J $(GRID_DIM) $(POINTS) $(HULL) $(THREADS)

# ----------
# GRAHAM
# ----------

g: requiredFolders compile runG

runG:
	./$(MAIN) -G $(GRID_DIM) $(POINTS) $(HULL) $(THREADS)

# ----------
# QUICKHULL
# ----------

qh: requiredFolders compile runQH

runQH:
	./$(MAIN) -QH $(GRID_DIM) $(POINTS) $(HULL) $(THREADS)

# ----------
# QUICKHULL WITH CRAWLERS
# ----------

qhc: requiredFolders compile runQHC
	
runQHC:
	./$(MAIN) -QHC $(GRID_DIM) $(POINTS) $(HULL) $(THREADS)

# ----------
# CHAN
# ----------

ch: requiredFolders compile runCH

runCH:
	./$(MAIN) -CH $(GRID_DIM) $(POINTS) $(HULL) $(THREADS)

# ----------
# CONCURRENT
# ----------

co: requiredFolders compile runCO

runCO:
	./$(MAIN) -CO $(GRID_DIM) $(POINTS) $(HULL) $(THREADS)
