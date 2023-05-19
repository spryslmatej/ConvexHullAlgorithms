OUT = out
BIN = bin
OBJ = obj

REQ_FOLDERS = $(OUT) $(OBJ) $(BIN) $(OBJ)/generators

requiredFolders:
	mkdir -p $(REQ_FOLDERS)

clear:
	rm -rf $(REQ_FOLDERS)

POINTS = points.data

GPP_FLAGS = -Wall -pedantic -O3 -std=c++11

GEN_POINT_COUNT = 100000000
GEN_MODE = 3

# GEN_MAX_VAL = 100
 
# long long int
# GEN_MAX_VAL = 9223372036854775807

# int
GEN_MAX_VAL = 2147483647


OBJS= \
	$(OBJ)/main.o \
	$(OBJ)/generators/clusterGenerator.o \
	$(OBJ)/generators/randomClusterGenerator.o \
	$(OBJ)/generators/circleGenerator.o \
	$(OBJ)/generators/fuzzyCircleGenerator.o \

$(OBJ)/%.o: src/%.cpp
	g++ $(GPP_FLAGS) -c $< -o $@
	
compile: requiredFolders $(OBJS)
	g++ $(GPP_FLAGS) $(OBJS) -o $(BIN)/generator

generate: compile run

run:
	./$(BIN)/generator $(GEN_POINT_COUNT) $(GEN_MODE) $(GEN_MAX_VAL) $(OUT)/$(POINTS)
