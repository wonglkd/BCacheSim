CC=g++-11
CC_FLAGS=-O3

ALL: bin/benchmark bin/benchmark-dev bin/device-size

bin/%: %.cpp
	$(CC) $(CC_FLAGS) $^ -o $@
