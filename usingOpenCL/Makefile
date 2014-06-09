#
# Simple makefile for usingOpenCL example
# 
CC=nvcc
CFLAGS=-arch=sm_20
LFLAGS=-lOpenCL

all:
	$(CC) $(CFLAGS) usingOpenCL.cpp -o usingOpenCL $(LFLAGS)

clean:
	rm -rf *o usingOpenCL
