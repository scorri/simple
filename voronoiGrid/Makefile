#
# Makefile for simple example of using thrust library
#

CC=nvcc
CFLAGS=-arch=sm_20
LFLAGS=-lfreeimage

all:
	$(CC) $(CFLAGS) voronoiGrid.cu -o voronoiGrid $(LFLAGS)

clean:
	rm -rf *o *png voronoiGrid 
