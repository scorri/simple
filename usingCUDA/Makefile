#
# Simple makefile for increment array cuda example
# 
CC=nvcc
CFLAGS=-arch=sm_20

all:
	$(CC) $(CFLAGS) incrementArray.cu -o increment

clean:
	rm -rf *o increment
