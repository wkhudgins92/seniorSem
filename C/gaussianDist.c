// William Hudgins
// CSCI 437
// gaussianDist.c
// 01/23/2014
//
// Distributed gaussian Elimination in C

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define NUM_ROWS 30
#define NUM_COLS 31

int main(int argc, char **argv)
{
	// Set up MPI
	MPI_Init(&argc, &argv);
	MPI_Status status;
	int id, clusterSize;
	MPI_Comm_size(MPI_COMM_WORLD, &clusterSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	
	// Initialize key variables
	int i, j, k, l;
	double tmp, multiplier;

	// Calculate the number of row operations necessary
	int numRowOps, numRowOps2, maxRowOps;
	maxRowOps= 0;
	for (i = (NUM_ROWS - 1); i > 0; i--)
		maxRowOps += i;
	
	// These two variables are used to track remaining number of row operations for sending and receiving respectively
	numRowOps = maxRowOps;
	numRowOps2 = numRowOps;
		
	/// Code for Master
	if (id == 0)
	{
		// Initialize key variables
		int slaveDone, currentSlave, tag, index, pivotRow, iterationNum;
		int slaveCount[clusterSize - 1];
		int oldSlaveCount[clusterSize - 1];

		//double matrixA[4][5] = {{6, -2, 2, 4, 16}, {12, -8, 6, 10, 26}, 
		//	{3, -13, 9, 3, -19}, {-6, 4, 1, -18, -34}};
	double matrixA[NUM_ROWS][NUM_COLS] = {{1071, 55, 34, 87, 74, 6, 37, 37, 75, 72, 68, -6, -9, 83, 36, 70, -1, 35, 0, 38, -17, 12, -18, 37, 66, 27, 82, 4, 14, 17, 0},
   {84, 1261, 23, 18, 55, 14, -10, 45, 41, 76, 19, 62, 88, 65, -3, 84, 5, 49, -14, 60, 88, 31, 24, -12, 22, 88, 28, 23, 49, -7, 89},
   {1, 66, 1286, 36, 46, 40, 23, 20, 72, 15, 56, 47, 63, 23, 15, -19, 75, 59, 67, 65, 77, 13, 88, 57, 49, 87, -8, -1, 75, 47, 8},
   {42, 53, 86, 1179, 22, 29, 40, 6, 12, 90, -7, -9, 2, 37, 26, 75, 41, 39, 5, 46, 71, 59, 81, 87, 52, 11, -17, 59, 21, 60, -18},
   {45, 39, 59, 24, 1523, 65, 70, 37, 63, 10, -9, 57, 9, 32, 66, 30, 59, 20, 74, -4, 65, 61, 34, 42, 55, 79, 70, 62, 89, 80, 31},
   {65, 18, 41, -2, 11, 1219, 84, 21, -16, 35, 69, 68, 1, 89, -10, 58, 7, 16, 47, 28, 39, 39, 66, 77, 61, 76, -10, -1, 13, 80, 73},
   {35, 42, 56, -15, 61, 37, 1182, 8, 78, 71, 59, 3, 17, 27, 36, 63, 2, 70, 68, 89, 88, 47, 0, 2, -16, 19, 63, 17, 0, 64, -11},
   {82, -7, 74, 41, 0, 45, 85, 1054, 63, 83, -12, 80, 5, 56, 79, 33, -5, 18, 78, 10, 41, -2, 6, 4, -17, 2, -4, 86, 24, 35, 55},
   {76, 48, 60, 64, 82, 14, 37, 13, 1192, 45, 31, 42, 36, 4, 77, 0, 84, -16, 76, 8, 57, 68, 85, 85, 24, -16, 13, 53, 19, -13, -2},
   {-5, 59, 29, 33, 79, -6, -1, 37, 50, 995, 65, 6, 40, 13, 50, 20, 70, 24, -15, 1, 35, 89, -1, 24, 90, -1, 86, 17, 28, -13, 43},
   {28, 84, 29, 82, 71, 58, 42, 4, 69, 82, 1560, -15, 31, 56, 66, 87, -17, 72, 84, 44, 16, 75, 54, 41, 74, 69, 26, 80, -16, 27, 72},
   {79, -12, 64, 39, 35, -14, -7, 28, 88, 78, -9, 1097, 84, 38, 85, 0, -2, 83, 41, 72, -4, 46, -5, 72, 9, 19, 1, 73, 55, -11, 22},
   {88, 43, -19, 50, 50, 84, 66, 37, -13, 79, 60, 71, 1186, 46, 78, 5, 64, 0, 12, -14, 35, 76, -20, 47, 43, 5, 11, -15, 11, 42, 61},
   {85, 51, 89, 90, 28, -11, -14, 41, 66, 66, 30, 80, 25, 1397, 73, 23, 34, 38, -11, 18, 84, 83, 73, 5, 23, 70, 22, 19, 21, 27, 87},
   {53, 4, 17, -9, 90, 54, 7, 3, 39, 89, 56, 60, 17, -4, 1178, 77, 28, 34, 39, 70, 79, 87, 2, 21, 7, 71, 35, 1, 35, 73, 31},
   {-20, 33, 47, 60, 27, -20, -14, -18, 39, 30, 18, -16, -6, -14, 51, 841, 33, 89, 26, -10, 27, 39, 41, 29, 77, 85, 53, -13, 19, 67, 45},
   {58, -9, 55, 33, 14, 26, -1, 64, 87, 82, 1, 40, -16, 23, 54, 4, 1009, 8, 52, 76, 34, -4, -13, 69, 59, 58, 78, 6, 0, 23, 21},
   {15, 40, 63, 22, 56, -13, 4, 39, 43, -2, 6, 21, 57, 1, 89, 29, 67, 1045, 7, 51, 6, 47, 36, -5, -17, 85, 50, 21, 75, 50, 10},
   {40, 5, -17, 49, 83, 23, 8, 51, 34, 51, 85, -15, 19, -5, 12, -2, -18, 51, 688, -19, -6, 83, 5, 4, 4, -19, 83, 13, 0, -16, 46},
   {0, 20, 41, 5, 47, 77, -10, 6, 3, -20, -9, 90, 27, 69, 0, 3, 55, 9, 35, 1058, 58, 64, 69, 0, 32, 35, 45, 47, 77, 59, 28},
   {-18, 40, -19, 47, -16, 49, 25, 42, 45, 51, 52, 5, -17, 24, 37, 66, 82, 49, 19, 36, 1047, 10, 84, 50, 68, 4, 71, 6, 62, 18, 9},
   {83, 0, 56, -3, 0, 35, 49, -7, 69, 69, 50, 26, -9, 27, 61, -10, 39, -14, 76, 16, 1, 1037, 19, 12, 60, 27, 20, 76, 27, 11, 73},
   {3, 31, 52, 76, 89, 81, -1, 71, 69, 39, 2, 52, 19, -9, -6, 31, 72, 51, 78, 54, 58, 88, 1380, 32, 25, 37, 2, 1, 49, 77, 56},
   {30, 60, 16, 41, 60, -9, 24, 33, 53, 50, -8, 21, 75, 8, 38, 20, 36, -4, 31, 47, 79, 13, 77, 1054, 55, 9, 1, 18, 89, 86, -2},
   {81, 48, 13, 18, 88, 39, -16, 35, -9, 24, 89, -18, 18, 0, -10, 70, 33, 43, -12, 41, 75, 13, -8, 30, 987, -3, 19, 72, 46, 50, 48},
   {-18, 59, 7, 68, 19, 79, 20, 0, -18, 49, 26, 74, 67, 86, 8, 9, 78, 83, 40, 10, 34, 38, 84, 5, 87, 1188, 22, -14, 55, 10, 22},
   {11, 42, 6, 39, 51, 45, 0, 82, 50, 12, 38, 45, 59, -5, 66, -6, 19, 42, 80, 77, 70, 21, 46, 52, 41, -6, 1225, 52, 72, 16, 49},
   {19, 4, 29, 32, 89, 81, 2, 78, 57, 38, 58, 78, -16, -18, 63, 5, 74, 48, 90, 36, -7, 31, 56, -12, 19, 33, 74, 1152, -8, 30, 15},
   {62, 90, -18, 22, -2, 79, 77, 11, -19, 54, 23, 61, 34, -1, 29, 22, 32, 52, 8, 58, 40, 20, -11, 87, 32, 55, 73, 48, 1198, -16, 90},
   {11, 34, -3, 67, 60, 3, 70, 86, -9, 75, 74, 85, -3, 48, 74, 88, 2, 78, 73, -12, 83, 64, 60, 44, 70, 43, 45, 78, 60, 1585, 45}};
		// Output matrix
		printf ("Starting matrix\n");
		printf ("[\n");
		for (i = 0; i < NUM_ROWS; i++)
		{
			printf("[");
			for (j = 0; j < NUM_COLS; j++)	
			{
				if (i != NUM_ROWS - 1 && j != NUM_COLS)
					printf ("%.2f, ", matrixA[i][j]);
				else
					printf ("%.2f", matrixA[i][j]);
			}
			printf("]\n");
		}
		printf("]\n");
			
		// Set initial slave counts to 0
		for (i = 0; i < clusterSize - 1; i++)
		{
			slaveCount[i] = 0;
			oldSlaveCount[i] = 0;
		}
		
		// Begin forward elimination
		for (iterationNum = 0; iterationNum < (NUM_ROWS - 1); iterationNum++)
		{			
			// Find pivot
			pivotRow = iterationNum;
			for (i = iterationNum + 1; i < NUM_ROWS; i++)
				if (matrixA[i][iterationNum] > matrixA[pivotRow][iterationNum])
					pivotRow = i;

			// Swap rows
			for (j = iterationNum; pivotRow != iterationNum && j < NUM_COLS; j++)
			{
				tmp = matrixA[pivotRow][j];
				matrixA[pivotRow][j] = matrixA[iterationNum][j];
				matrixA[iterationNum][j] = tmp;
			}

			// These are reinitialized each iteration on purpose
			int rootIndex = 0;
			double tmpRow[NUM_COLS]; 
			MPI_Request request[(NUM_ROWS - (iterationNum + 1))][4];
			
			//  Modify each row
			// First send data to all slaves
			for (i = iterationNum + 1; i < NUM_ROWS; i++) 
			{
				numRowOps--; // Decrement number of operatins remaining
				currentSlave = (maxRowOps - numRowOps) % (clusterSize - 1); // Determine current slave
				if (currentSlave == 0) currentSlave = clusterSize - 1;
				
				// Check if slave is done with work
				if ((numRowOps + 1) <= (clusterSize - 1))
					slaveDone = 1;
				else 
					slaveDone = 0;
					
				slaveCount[currentSlave - 1]++; // Increment slave counter
				tag = currentSlave * currentSlave * (slaveCount[currentSlave - 1] + 1); // Determine basic tag
				if (currentSlave == 1) tag += 375;
				
				// Send pivot row, target row, iteration number, and slave status to slaves
				MPI_Isend(matrixA[iterationNum], NUM_COLS, MPI_DOUBLE, currentSlave, tag, MPI_COMM_WORLD, &request[rootIndex][0]);
				MPI_Isend(matrixA[i], NUM_COLS, MPI_DOUBLE, currentSlave, tag + 100, MPI_COMM_WORLD, &request[rootIndex][1]);
				MPI_Isend(&iterationNum, 1, MPI_INT, currentSlave, tag + 150, MPI_COMM_WORLD, &request[rootIndex][2]);
				MPI_Isend(&slaveDone, 1, MPI_INT, currentSlave, tag + 250, MPI_COMM_WORLD, &request[rootIndex++][3]);
			}
			
			// Wait for the messages to finish being sent
			index = 0; 
			MPI_Request allRequest[(NUM_ROWS - (iterationNum + 1)) * 4];
			for (i = 0; i < (NUM_ROWS - (iterationNum + 1)); i++) // Flatten nested array
				for (j = 0; j < 4; j++)
					allRequest[index++] = request[i][j];
			MPI_Status allStatus[index];
			MPI_Waitall(index, allRequest, allStatus);
			
			// Receive modified rows back from slaves
			for (i = iterationNum + 1; i < NUM_ROWS; i++) 
			{
				numRowOps2--;
				currentSlave = (maxRowOps - numRowOps2) % (clusterSize - 1);
				if (currentSlave == 0) currentSlave = clusterSize - 1;
				oldSlaveCount[currentSlave - 1]++;
				tag = currentSlave * currentSlave * (oldSlaveCount[currentSlave - 1] + 1);
				if (currentSlave == 1) tag += 375;

				MPI_Recv(tmpRow, NUM_COLS, MPI_DOUBLE, currentSlave, tag + 200, MPI_COMM_WORLD, &status);				

				for (j = iterationNum; j < NUM_COLS; j++)
					matrixA[i][j] = tmpRow[j];
			}
		}

		// Perform back substitution here
		double* solutionSet = malloc(sizeof(double) * NUM_ROWS);	
		for (i = NUM_ROWS - 1; i >= 0; i--)
		{
			for (j = NUM_ROWS - 1; j > i; j--)
				matrixA[i][NUM_COLS - 1] -= matrixA[i][j] * solutionSet[j];
			solutionSet[i] = matrixA[i][NUM_COLS - 1] / matrixA[i][i];
		} 

		printf("]\n");
		printf ("Solution set:\n");
		printf ("[");
		for (i = 0; i < NUM_ROWS; i++)
			printf("%.2f, ", solutionSet[i]);
		printf("]\n"); 
		//exit(0); // Exit, terminating the program
	}

	// Slave code
	else if (id > 0)
	{
		// Initialize key variables
		int tag, iterationNum, done, worked = 1, index = 0;
		double multiplier;
		double pivotRow[NUM_COLS];
		double targetRow[NUM_COLS];
		MPI_Request fakeRequest[NUM_ROWS + 1]; // We don't always know the needed size of the array, so this is used as a placeholer
		
		// In case we have more slaves than row operations
		if (id < maxRowOps)
			done = 0;
		else 
		{
			done = 1;
			worked = 0;
		}
			

		while (done != 1)
		{
			// Determine tag
			tag = id * id * (index + 2);
			if (id == 1)
				tag += 375;
				
			// Receive pivot row, target row, iteration number, and status from master
			MPI_Recv(pivotRow, NUM_COLS, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
			MPI_Recv(&iterationNum, 1, MPI_INT, 0, tag + 150, MPI_COMM_WORLD, &status);
			MPI_Recv(targetRow, NUM_COLS, MPI_DOUBLE, 0, tag + 100, MPI_COMM_WORLD, &status);
			MPI_Recv(&done, 1, MPI_INT, 0, tag + 250, MPI_COMM_WORLD, &status);

			// Modify target row
			multiplier = targetRow[iterationNum]/pivotRow[iterationNum]; 
			for (j = iterationNum; j < NUM_COLS; j++)
				targetRow[j] -= pivotRow[j] * multiplier;
			
			// Send modified target row to master
			MPI_Isend(targetRow, NUM_COLS, MPI_DOUBLE, 0, tag + 200, MPI_COMM_WORLD, &fakeRequest[index++]); 
		}
		
		if (worked != 0)
		{
			// Wait for messages to be sent
			MPI_Request request[index]; // Put requests in a smaller array
			for (i = 0; i < index; i++)
				request[i] = fakeRequest[i];
			MPI_Status allStatus[index];
			//MPI_Waitall(index, request, allStatus);
		}
		MPI_Finalize(); // Slaves can exit here
	}
		

	return 0;
}

