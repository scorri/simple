#
# Makefile for simple example of using thrust library
#

CC=nvcc
CFLAGS=-arch=sm_20

all:
	$(CC) $(CFLAGS) usingThrust.cu -o usingThrust

clean:
	rm -rf *o usingThrust
