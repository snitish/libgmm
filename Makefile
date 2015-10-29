CC ?= gcc
CFLAGS = -std=c99 -Wall -O3 -fPIC
AR ?= ar

all: test lib

lib: gmm.o
	$(AR) rcs libgmm.a gmm.o

test: test.c gmm.o
	$(CC) $(CFLAGS) test.c gmm.o -o test -lm

gmm.o: gmm.c gmm.h
	$(CC) $(CFLAGS) -c gmm.c -lm

clean:
	rm -f gmm.o libgmm.a test
