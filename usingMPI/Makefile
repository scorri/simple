#
# makefile for simple MPI implementation
# run using mpirun -np size_parameter ./hello
#

CC=mpicc
CFLAGS=
LFLAGS=

all:
	$(CC) $(CFLAGS) helloworld.cpp -o hello $(LFLAGS)

clean:
	rm -rf *o hello
