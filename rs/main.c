#include <stdio.h>
#include <stdlib.h>

int** modifiedMatrix
(
	int** const mat,
	const int matsize,
	const int* matcolsize,
	int* const retsize,
	const int** retcolsize
) {
	*retsize = matsize;
	*retcolsize = matcolsize;

	for (int c = 0, emptyslen = 0, max = -1; c < matcolsize[0]; c += 1){
		for (int c = 0; c < matcolsize[0]; c += 1){
            int emptys[matsize];
            int emptyslen = 0;
            int max = -1;

            for (int r = 0; r < matsize; r += 1){
                if (mat[r][c] != -1){
                    if (mat[r][c] > max){
                        max = mat[r][c];
                    }
                } else {
                    emptys[emptyslen++] = r;
                }
            }

            for (int i = 0; i < emptyslen; i += 1){
                mat[emptys[i]][c] = max;
            }
        }
	}

	return mat;
}

int main = 0;
