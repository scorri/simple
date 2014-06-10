/*
	A simple MPI example - hello world
*/

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv)
{
	int size, rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	printf("Hello from process - rank(%d) of size(%d)\n", rank, size);
	MPI_Finalize();

	return 0;
}
