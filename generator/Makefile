CC = cc
CFLAGS = -O2 -Wall
LDFLAGS = -lm

# Minimal source files for Bridges puzzle generation
SRCS = bridges_gen.c bridges.c dsf.c findloop.c malloc.c misc.c nullfe.c random.c

all: bridges_gen

bridges_gen: $(SRCS) puzzles.h
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f bridges_gen *.o

.PHONY: all clean

