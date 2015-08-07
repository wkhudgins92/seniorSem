// William Hudgins
// CSCI 437
// jacobi.c
// 11/03/2014
//
// Distributed Jaocbi in C

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#define NUM_ROWS 45
#define NUM_COLS 46

void jacobi(double matrixA[NUM_ROWS][NUM_COLS ], double solutionSet[NUM_ROWS], MPI_Status status, int clusterSize);
double array_sum(double arr[], int len);
void generateEquations(double matrixA[NUM_ROWS][NUM_COLS ], double equations[NUM_ROWS][NUM_COLS ], MPI_Status status, int clusterSize);
void plugin(double estimate[NUM_ROWS], double equations[NUM_ROWS][NUM_COLS ], double newEstimate[NUM_ROWS]);
void generateEquation(int id, MPI_Status status, int clusterSize);

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	MPI_Status status;
	int i, j, k, id, clusterSize;
	MPI_Comm_size(MPI_COMM_WORLD, &clusterSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
		double matrixA[NUM_ROWS][NUM_COLS] = {{1762, 24, 67, -6, 9, 28, 43, 70, 38, 65, 81, 2, 70, 62, 52, 20, 44, 80, 60, 59, 46, -19, 41, 48, 12, 44, 19, -9, -4, -3, 60, -6, 48, -18, 22, 79, 51, 38, -4, 65, 63, 40, 36, 67, 86, 78},
   {43, 1615, 46, -18, -15, 81, 77, 54, 72, -6, -11, 9, 48, 63, -9, 25, 84, -1, -17, 66, 81, 0, 51, 76, 27, 37, -1, 14, 87, 69, 67, 28, 67, 25, 69, 26, -4, -19, 74, -12, -5, 18, 81, 15, 9, 74},
   {14, 74, 1723, 45, -6, 40, -5, 36, 86, 8, 59, 60, 55, 52, 33, 62, 2, -2, 68, 44, 17, 88, 34, 3, 16, 30, 65, 22, 39, 43, 53, 62, 20, -9, 42, 18, 53, 51, 68, 48, 81, -3, 79, 33, -2, 32},
   {50, 15, 1, 1388, 37, 31, 8, 4, 22, 20, -10, 52, -3, 74, -19, 16, -4, -2, 38, -9, 51, 16, 64, 0, 35, -8, 87, 18, 90, -1, 35, -8, 70, 65, 80, -7, 17, 42, 85, 15, 58, 35, 33, 86, 61, 26},
   {69, 24, -15, 89, 1508, -5, 3, 14, -13, -2, -1, 16, 16, 14, 1, 89, 89, 48, 18, 47, 26, 45, 1, 35, -8, -11, 68, -12, 31, 77, 45, 89, 47, -2, 4, 6, 3, 47, 21, 66, 13, 90, 77, 63, 88, 57},
   {-15, 49, 76, 69, 54, 1816, 71, 83, 55, 10, 82, 61, 48, 8, 67, 56, -13, 75, 73, 85, -3, -9, -10, 25, 63, 29, 45, 47, 23, 0, 60, 76, 85, 59, -16, 28, -11, 89, -5, 51, -4, 32, 9, 46, 34, 63},
   {30, 87, 34, 32, -17, 73, 1854, 38, -11, 67, -2, 55, 60, 89, 21, 75, 78, 41, 79, 53, -10, -10, 39, 86, 3, 50, 30, -1, 33, 34, 61, 53, 15, 58, 72, 40, 59, 32, 39, 26, -11, 70, 68, 64, 10, 23},
   {20, 71, 55, 43, -8, 72, 47, 2110, 30, 70, 89, 57, 8, 7, -5, 47, -14, 71, 15, 86, 57, 17, 89, 74, -13, 32, 16, 89, 53, 58, 85, 16, 59, 87, 69, 72, 82, 31, 3, 17, 28, 32, 31, 27, 65, 87},
   {54, 43, 39, 86, 67, 45, 9, 74, 1960, -15, 74, 29, 78, 1, 87, 67, 56, 47, 56, 63, 21, -11, -3, 6, 26, 67, 66, 32, 44, 46, -7, 39, 85, 57, 7, 39, -16, 14, 54, 32, 84, 51, 72, 35, 44, 33},
   {6, 0, 47, 16, 2, 17, -17, 59, 88, 1377, 33, 15, -15, 52, 90, 54, 9, 17, 50, 26, -10, 18, 43, 89, -20, -5, -6, 28, 1, 10, 56, 35, -13, 48, 51, 54, 26, 46, 65, 3, 5, 32, 63, 80, 71, -14},
   {81, 90, 22, 29, 9, 46, -5, 46, 15, 36, 1313, 12, 24, 1, 9, 8, 8, 78, -11, 15, 60, 21, 48, 6, 12, 48, 18, 7, 32, 88, -11, 81, 47, 39, 49, 45, -4, 0, 86, -9, -17, 44, 23, 32, 4, 39},
   {82, -8, 55, 76, 73, 59, 26, 57, 80, 49, 45, 1871, -9, 22, 87, 25, 11, 58, 10, 49, -2, 77, 16, 79, 57, 7, 25, 62, 89, 67, 70, -15, -1, 59, 45, -19, 60, 35, 40, 43, 7, 39, 54, 53, 26, 41},
   {75, 17, -1, 30, 78, 87, 76, 82, 27, 48, 6, 55, 1624, -3, 3, 53, 50, 42, -14, 87, 30, 84, 41, 56, -10, 60, 34, 11, 42, -10, 63, -3, 80, -13, 35, 39, 25, 10, 41, 18, 18, -3, 8, -2, 58, 18},
   {81, 27, 2, 10, 12, -10, 39, 6, 37, 36, 23, 48, 31, 1725, 87, 79, 46, 2, 62, 19, 37, 68, 43, 7, 31, 5, 83, 45, 43, 18, 63, -14, 69, 47, 11, 38, 58, 90, 86, 45, 25, 3, 82, 29, -2, 71},
   {0, 38, 88, 40, 41, 28, -10, 18, 13, 4, -17, 19, 8, -20, 1487, 11, 75, 54, 59, 5, 42, 31, 33, 48, 59, -8, -18, 70, 13, 35, 34, 40, 38, 56, 58, 81, 90, 53, 49, 55, 0, 7, 73, 0, 7, 15},
   {23, -3, 42, 45, 77, 20, 3, 47, 3, 63, 9, 87, 48, 54, 69, 1670, 1, -6, 69, 66, -2, -13, 8, 45, 29, 22, 22, 47, 10, 29, 48, 74, 42, -3, 15, 81, 84, 69, -12, 72, 2, 25, 45, 73, 43, 56},
   {-8, -3, 11, 16, 58, 39, 30, 37, -8, 78, 39, 10, 52, 57, 44, 26, 1450, -11, 28, 53, 47, 43, 74, -9, 8, -7, -8, 31, 86, -14, 41, 35, 76, -15, 13, 61, 64, 68, -20, 24, 5, 82, 52, 34, 15, 85},
   {85, 9, 33, 84, 47, 68, -4, 87, 17, -3, -20, 65, 29, 18, -13, -6, 13, 1556, 72, 51, 0, 65, 86, 78, 5, 45, 70, 33, 9, 28, -16, 35, 38, -20, 51, -16, 78, 33, 37, 42, 88, 84, -3, 7, 49, 13},
   {29, -20, 83, -10, 40, 47, 56, 82, 27, 89, 7, 22, 4, -17, 89, 63, -7, 33, 1707, 31, 28, 59, 44, 50, 70, 10, 61, 8, 90, 24, 90, 58, 83, -11, -8, -5, 31, 4, 87, 19, 64, 80, 19, 29, -4, 45},
   {5, 14, 45, 1, 70, 66, 33, -10, 40, 57, 85, 54, 4, -4, 9, 83, 76, -3, 21, 1811, 69, 26, 63, 7, 38, 56, 43, 2, 8, 36, 63, 78, -7, 54, 30, 75, 43, -1, 13, 17, 80, 62, -12, 68, 81, 73},
   {26, 38, 49, -7, 28, 88, 4, 1, 86, 55, -4, 9, -4, -11, 58, 63, 80, 68, 37, 25, 1478, 65, 77, -1, -18, 78, -14, -10, 14, 35, 56, 66, 8, 40, 66, 49, -18, 2, 17, 57, -10, 12, -6, 73, -7, 54},
   {39, 77, 21, 30, 84, 12, 24, 75, 30, 45, 64, -9, 35, 90, 15, 35, 28, 38, -3, -7, 87, 1622, 16, 51, 53, 28, 90, 64, 14, 9, 14, 1, 21, 29, 45, 37, -17, 23, 6, 56, 48, 54, 9, -11, 61, 13},
   {38, 88, 37, 56, 23, 20, 8, 14, 9, 53, 12, -10, 62, 81, 65, 37, 2, 86, 42, 30, 5, -15, 1810, 52, 22, 51, 25, 72, 10, 22, 19, 8, 80, 80, -13, 28, 45, 62, 73, 41, 38, 61, 24, 26, 78, 72},
   {60, 19, 4, 84, 31, 7, 20, 40, 26, 77, 82, 23, 53, 42, 29, 29, -18, 28, 53, 38, -5, -19, 25, 1667, 44, 86, -13, 53, 30, -17, 81, 51, -15, 49, 19, 88, 73, 70, 53, 49, 32, -20, 2, 87, 62, 13},
   {41, 12, 51, 83, 82, 49, -10, 49, 18, -9, -15, 28, 6, 47, 10, -7, 31, 16, 38, 11, 10, 30, 6, 7, 1298, 81, -4, 38, 22, 45, 70, 53, 40, 53, 42, -20, 66, -7, 25, 24, 39, 31, -5, -20, 5, 82},
   {90, 51, -16, -9, 81, -14, 68, -12, 45, 55, 20, 37, 74, 25, 33, 33, 13, 77, -17, 74, 51, 8, 60, -19, 58, 1451, 53, 41, 77, -12, -10, -3, 72, 73, 76, -10, 11, -9, 28, 23, 62, 88, -2, 5, -8, -9},
   {39, 3, 15, 38, -10, 2, 29, 57, -8, 16, 64, 71, 33, 88, 77, 79, 35, 24, -14, 33, 21, 19, 49, 63, 8, 29, 1607, 86, 10, 56, 90, 36, 41, 69, -14, 80, 6, -11, 64, 77, 23, 14, -7, 27, 42, 3},
   {-12, 9, 1, 19, -11, 40, 17, 40, -10, 31, -5, 20, -12, 13, 24, 16, 55, 50, -4, 12, -6, 32, -3, -16, 51, 17, 32, 1005, -5, -8, -16, 54, 13, 55, 75, 80, 28, 59, -17, 42, 18, 50, 50, -17, 67, 71},
   {84, 8, 61, -15, 64, 74, 25, -1, 3, 44, 11, 21, 78, 24, 49, 4, 65, 35, -15, -13, 31, 25, 59, -14, -9, 25, 4, 34, 1286, -9, 20, 15, 36, 11, 90, 33, 7, 27, 14, -4, 31, 58, 21, 2, 68, 39},
   {35, 0, 71, 19, 13, 90, 74, -17, 71, 55, -15, -10, 78, 68, -13, 11, 66, 71, -20, 43, -11, -3, 40, 47, -19, 3, -3, 54, -20, 1550, 26, 77, 2, 13, 32, 77, 85, -5, 49, -7, 73, 66, 78, 62, 6, 62},
   {82, 55, 36, 63, 34, -8, 44, 71, 15, 90, 5, 57, -6, 33, 87, -18, 17, 7, 9, 49, -6, 14, 57, 9, 29, -10, 9, -7, 85, 63, 1446, -14, -1, 14, -9, 49, 71, 21, 46, 49, 65, 2, 68, 24, 12, 0},
   {17, 87, -16, 23, 32, 89, 74, 28, 19, 4, 86, 56, -3, 90, 1, -11, 89, -11, 15, 12, 59, 68, 44, 42, -4, 48, 73, -4, 55, 7, 17, 1575, 0, 34, -11, 55, 40, 9, 57, 12, 83, 9, 47, -16, 25, 51},
   {54, 12, 78, 60, 26, -1, 43, 37, -1, 23, 54, 26, 69, 60, 70, 50, 46, 40, 57, 42, 72, 15, 83, 18, 79, -1, -10, -14, 13, -17, 48, 47, 1501, 11, 63, 38, -12, 16, 37, -12, 38, 78, -1, -15, 50, 17},
   {8, -20, 49, 6, 30, 36, 70, 61, 57, 42, 7, -10, 72, 10, 1, 38, 6, -6, 63, 71, 32, 49, 84, -7, 56, 57, -14, 42, 79, 74, 69, -12, 0, 1432, 2, 68, 3, 53, -15, -14, 74, 34, 12, 74, -17, -16},
   {66, 3, 78, 51, 90, -12, 86, 24, 78, -14, 14, 70, 89, 64, 44, 42, 5, 7, 90, 74, 18, 31, 24, 33, 22, 72, 28, 28, 83, -12, 85, 14, 43, 13, 1988, 38, 65, 29, 17, 37, 78, 61, 57, 33, 50, 17},
   {76, 55, 61, 9, 43, 52, 20, 55, 50, 77, 37, 39, 37, 50, 83, 85, 67, 73, 75, 47, 78, 17, 31, 84, 88, 67, 48, 34, 18, 19, 65, 19, -10, 57, 47, 2162, 42, 18, 11, 85, 26, 61, 74, 36, -6, 53},
   {46, 75, 68, 39, 66, 53, 47, 56, 24, 63, 65, 86, 16, -8, 24, 54, -15, 88, 12, 25, 62, 29, 63, 24, 54, 21, 47, 38, 89, 85, -1, 80, 46, -19, 49, -11, 1944, 40, 76, 37, 33, 15, -6, 35, 46, 36},
   {78, 37, 88, 44, 53, 40, 73, -18, -6, 58, 60, 85, 39, -6, -12, 89, 8, 12, 73, -19, 41, 46, 38, 0, 51, 55, -10, 62, 57, -6, 90, 1, -6, -9, 65, 22, 69, 1609, 28, 64, -4, -4, 58, -12, 65, 57},
   {60, 10, 22, -20, -20, 25, 60, 6, 62, 78, 21, 83, 56, 12, 47, 32, 63, 11, 64, 45, 11, -17, 6, 64, 80, 75, 85, 15, -14, 37, 0, 1, 41, 60, 26, 75, 58, 23, 1588, -8, 42, -20, 63, 80, 37, -13},
   {61, 53, -2, 20, 2, 8, 47, 39, 35, -4, -7, -14, 0, 7, -13, 76, -2, 70, -1, 14, 11, 84, -5, 0, 43, 33, 56, 1, 55, 70, -17, -4, 55, 10, 48, 45, 83, 51, 11, 1149, 6, 33, 31, 4, -3, 22},
   {84, 21, 36, -4, 68, 33, 71, -4, 52, 82, 58, 46, -4, -9, 1, 87, 82, 22, 72, 59, 62, 74, -18, 89, 39, 83, 62, 68, 33, 26, -12, 21, 52, 82, 63, 61, 69, 45, 73, 19, 2156, 85, -16, 75, 76, 54},
   {11, 36, 69, 19, 42, 19, 12, 87, 79, 31, 69, 37, -13, 48, -18, 58, 18, 81, -11, 19, -8, 6, -11, 68, 12, 3, 47, 63, 27, 74, 77, 0, -19, 89, 60, 80, 62, 31, 53, 15, -4, 1531, 45, -3, 22, 15},
   {-9, 52, 61, 76, -19, 62, -1, 46, 83, 7, -18, -3, 78, 48, 14, 83, -12, 74, -9, -9, 68, 50, 5, -12, 73, 58, 64, 32, -19, 44, 42, 90, -20, 19, 61, 66, 42, 66, -6, 78, 46, 14, 1682, 21, 46, 50},
   {87, 77, 59, 65, 13, 88, 11, 26, 65, 54, 30, -20, 67, -7, 61, 44, 46, 70, 28, 49, 80, 59, 57, 20, 39, -16, 18, 78, 72, 24, 83, -7, 47, -5, -19, -8, 82, 63, 16, 48, 46, 69, 3, 1851, 1, -5},
   {35, 49, 87, 78, 75, 88, 43, 9, 21, 35, 32, 34, 45, 57, -11, 28, 55, 45, 75, 89, 42, -10, 9, 57, 8, 36, -1, 56, 55, 2, 16, 42, 66, 60, 47, 59, -18, 40, 15, 23, 28, 15, 41, 29, 1853, 73}};   	//double matrixA[NUM_ROWS][NUM_COLS ] = {{2, -1, 0, 1}, {-1, 3, -1, 8}, {0, -1, 2, -5}};

	if (id == 0)
	{
		printf("Starting matrix\n");
		printf("[\n");
		for (i = 0; i < NUM_ROWS; i++)
		{
			printf("[");
			for (j = 0; j < NUM_COLS ; j++)
				printf("%.2f, ", matrixA[i][j]);
			printf("]\n");
		}
		printf("\n]\n");
		double solutionVector[NUM_ROWS] = {0}; 
		jacobi(matrixA, solutionVector, status,clusterSize);
		printf("Solution estimate:\n");
		printf("[");
		for (k = 0; k < NUM_ROWS; k++)
		{
			printf("%.2f, ", solutionVector[k]);
			if (k == NUM_ROWS - 1)
			{	
				printf("]\n");	
			}
		}
		printf("\nDONE\n");
		exit(0);
		
	}	

	else if (id > 0)
		generateEquation(id, status, clusterSize);

	MPI_Finalize();
	
	return 0;
}

void jacobi(double matrixA[NUM_ROWS][NUM_COLS], double solutionVector[NUM_ROWS], MPI_Status status, int clusterSize)
{
	int i, j;
	double TOLERANCE = 0.00000001;
	double equations[NUM_ROWS][NUM_COLS];
	generateEquations(matrixA, equations, status, clusterSize);
	double newEst[NUM_ROWS];
	double oldEstimate[NUM_ROWS] = {0};
	plugin(oldEstimate, equations, newEst);

	while (fabs(array_sum(oldEstimate, NUM_ROWS) - array_sum(newEst, NUM_ROWS)) > TOLERANCE)
	{
		for (i = 0; i < NUM_ROWS; i++)
			oldEstimate[i] = newEst[i];
		plugin(oldEstimate, equations, newEst);
	}
	
	for (i = 0; i < NUM_ROWS; i++)
		solutionVector[i] = newEst[i];
	return;
}

double array_sum(double arr[], int len)
{
	int i;
	double sum = 0;
	for (i = 0; i < len; i++)
		sum += arr[i];
	return sum;
}

void generateEquation(int id, MPI_Status status, int clusterSize)
{
	double row[NUM_COLS];
	int i, k, diag, tag;
	int rowNum, myWork;
	rowNum = 0;
	int numSlaves = clusterSize - 1;
	
	if (NUM_ROWS >= numSlaves)
	{
		int workForAllSlaves = NUM_ROWS / numSlaves;
		int extraSlavesNeeded = NUM_ROWS % numSlaves;
		myWork = workForAllSlaves;

		if (extraSlavesNeeded > 0)
		{
			int extraWorkTeamCaptain = workForAllSlaves + extraSlavesNeeded; // This slave will be the captain of a work crew that works OT
			if (id == extraWorkTeamCaptain)
				myWork++;
			else if (extraSlavesNeeded > 1 && ((extraWorkTeamCaptain - id) < (extraWorkTeamCaptain - 1)) && ((extraWorkTeamCaptain - id) > 0))
				myWork++;
		}
	}
	
	else 
	{
		if (id > NUM_ROWS)
			myWork = 0;
		else
			myWork = 1;
	}

	MPI_Request request[myWork];
	MPI_Status statusArr[myWork];
	for (k = 0; k < myWork; k++)
	{
		MPI_Recv(row, NUM_COLS, MPI_DOUBLE, 0, id, MPI_COMM_WORLD, &status);
		MPI_Recv(&rowNum, 1, MPI_INT, 0, id + (50 * (k + 1)), MPI_COMM_WORLD, &status);

		diag = row[rowNum];
		row[rowNum] = 0;
		for (i = 0; i < NUM_COLS; i++)
		{
			if (i != NUM_ROWS)
				row[i] = (row[i] / diag) * -1;	
			else if (i = NUM_ROWS)
				row[i] /= diag;
		}

		MPI_Isend(row, NUM_COLS, MPI_DOUBLE, 0, id + (100 * (rowNum + 1)), MPI_COMM_WORLD, &request[k]);
	}
	
	MPI_Waitall(k, request, statusArr);

	return;
}

void generateEquations(double matrixA[NUM_ROWS][NUM_COLS], double equations[NUM_ROWS][NUM_COLS], MPI_Status status, int clusterSize)
{
	int i, j, currentSlave, tag, index;
	int numSlaves = clusterSize - 1;
	double diag;
	double tmpRow[NUM_COLS];
	int rowNum[NUM_ROWS];
	
	if (numSlaves > NUM_ROWS)
		numSlaves = NUM_ROWS;
	
	for (i = 0; i < NUM_ROWS; i++)
	{
		rowNum[i] = i;
		for (j = 0; j < NUM_COLS; j++)
			equations[i][j] = matrixA[i][j];		
	}
	
	MPI_Request request[NUM_ROWS * 2];
	MPI_Status statusArr[NUM_ROWS * 2];
	index = 0;
	
	int slaveCount[clusterSize - 1];
	for (i = 0; i < clusterSize - 1; i++)
		slaveCount[i] = 0;
			
	// Send all data
	for (i = 0; i < NUM_ROWS; i++)
	{
		currentSlave = ((NUM_ROWS - i) % numSlaves) + 1;
		slaveCount[currentSlave - 1]++;
		MPI_Isend(matrixA[i], NUM_COLS, MPI_DOUBLE, currentSlave, currentSlave, MPI_COMM_WORLD, &request[index++]);
		MPI_Isend(&rowNum[i], 1, MPI_INT, currentSlave, currentSlave + (50 * slaveCount[currentSlave - 1]), MPI_COMM_WORLD, &request[index++]);
	}
	
	// Wait to make sure it all got sent
	MPI_Waitall(index, request, statusArr);

	// Receive data back
	for (i = 0; i < NUM_ROWS; i++)
	{
		currentSlave = ((NUM_ROWS - i) % numSlaves) + 1;
		MPI_Recv(tmpRow, NUM_COLS, MPI_DOUBLE, currentSlave, currentSlave + (100 * (i + 1)), MPI_COMM_WORLD, &status);
		for (j = 0; j < NUM_COLS; j++)
			equations[i][j] = tmpRow[j];
	}
	
	return;
}

void plugin(double estimate[NUM_ROWS], double equations[NUM_ROWS][NUM_COLS], double newEstimate[NUM_ROWS])
{
	int i, j;
	double sum;

	for (i = 0; i < NUM_ROWS; i++)
	{
		sum = 0;
		for (j = 0; j < NUM_COLS; j++)
		{
			if (j < NUM_ROWS)
				sum += estimate[j] * equations[i][j];
			else if (j == NUM_ROWS)
				sum += equations[i][j];
		}
		newEstimate[i] = sum;
	}
	return;
}	
