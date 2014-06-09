#
# Makefile for simple example which creates an image of 
# an evenly spaced voronoi grid
#

CC=nvcc
CFLAGS=-arch=sm_20
LFLAGS=-lfreeimage

all:
	$(CC) $(CFLAGS) voronoiGrid.cu -o voronoiGrid $(LFLAGS)

clean:
	rm -rf *o *png voronoiGrid 
