

/*
    extern __shared__ _FTYPE_ As[];
    //extern __shared__ _FTYPE_ Bs[];

    int block_x = blockIdx.x; int block_y = blockIdx.y;
    int thread_x = threadIdx.x; int thread_y = threadIdx.y;
    int block_dim_x = blockDim.x; int block_dim_y = blockDim.y;
    int grid_dim_x = gridDim.x; int grid_dim_y = gridDim.y;

    int I =  block_y * block_dim_y + thread_y;
    int J =  block_x * block_dim_x + thread_x;

    //printf("block_x %d thread_x %d block_y %d thread_y %d I %d J %d \n",block_x,thread_x,block_y,thread_y,I,J);
    //printf("Thread Dim x %d y %d Block Dim %d %d \n",block_dim_x,block_dim_y,block_dim_x,block_dim_y);
    int npad = ceil(N/((float) TILEDIM_K * 2)) * TILEDIM_K * 2;
    if((I < (npad/2)) && (J < (npad/2))){
        _FTYPE_ _c_n0 = 0;
        _FTYPE_ _c_n1 = 0;
        _FTYPE_ _c_m0 = 0;
        _FTYPE_ _c_m1 = 0;
        for(unsigned int c_tile = 0; c_tile < npad/TILEDIM_K; c_tile++){
            if(I < N && (c_tile*TILEDIM_K + thread_x) < N)
                As[thread_y*block_dim_y+thread_x] = A[I*N + c_tile*TILEDIM_K + thread_x];
            else
                As[thread_y*block_dim_y+thread_x] = 0.0;
            if((I+(npad/2) < N) && (c_tile*TILEDIM_K + thread_x < N))
                As[block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x] = A[(I+(npad/2))*N + c_tile*TILEDIM_K + thread_x];
            else
                As[block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x] = 0.0;
            if((c_tile*TILEDIM_K + thread_y) < N && J < N)
                As[2*block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x] = B[(c_tile*TILEDIM_K + thread_y)*N + J];
            else
                As[2*block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x] = 0;
            if(((c_tile*TILEDIM_K + thread_y) < N) && (J + (npad/2) < N))
                As[3*block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x] = B[(c_tile*TILEDIM_K + thread_y)*N + J + (npad/2)];
            else
                As[3*block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x] = 0.0;
            __syncthreads();
            
            //printf("before block_x %d block_y %d thread_x %d thread_y %d I %d J %d c_tile %d As [%d] %f [%d] %f [%d] %f [%d] %f A [%d,%d] %f [%d,%d] %f B [%d,%d] %f [%d,%d] %f \n",block_x,block_y,thread_x,thread_y,I,J,c_tile,thread_y*block_dim_y+thread_x,As[thread_y*block_dim_y+thread_x],block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x,As[block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x],2*block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x,As[2*block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x],3*block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x,As[3*block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x],I,c_tile*TILEDIM_K + thread_x,A[I*N + c_tile*TILEDIM_K + thread_x],I+(npad/2),c_tile*TILEDIM_K + thread_x,A[(I+(npad/2))*N + c_tile*TILEDIM_K + thread_x],(c_tile*TILEDIM_K + thread_y), J,B[(c_tile*TILEDIM_K + thread_y)*N + J],(c_tile*TILEDIM_K + thread_y), J + (npad/2),B[(c_tile*TILEDIM_K + thread_y)*N + J + (npad/2)]);

            for (unsigned int k = 0; k < TILEDIM_K; k++) {
                _c_n0 += As[thread_y*block_dim_y+k] * As[2*block_dim_x*block_dim_y + k*block_dim_y+thread_x];
                _c_n1 += As[thread_y*block_dim_y+k] * As[3*block_dim_x*block_dim_y + k*block_dim_y+thread_x];
                _c_m0 += As[block_dim_x*block_dim_y + thread_y*block_dim_y+k] * As[2*block_dim_x*block_dim_y + k*block_dim_y+thread_x];
                _c_m1 += As[block_dim_x*block_dim_y + thread_y*block_dim_y+k] * As[3*block_dim_x*block_dim_y + k*block_dim_y+thread_x];
                //printf("block_x %d block_y %d thread_x %d thread_y %d I %d J %d c_tile %d k %d As [%d] %f [%d] %f Bs [%d] %f [%d] %f c_n0 %f c_n1 %f c_m0 %f c_m1 %f \n",block_x,block_y,thread_x,thread_y,I,J,c_tile,k,thread_y*block_dim_y+k,As[thread_y*block_dim_y+k],block_dim_x*block_dim_y + thread_y*block_dim_y+k,As[block_dim_x*block_dim_y + thread_y*block_dim_y+k],2*block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x,As[2*block_dim_x*block_dim_y + thread_y*block_dim_y+thread_x],3*block_dim_x*block_dim_y + k*block_dim_y+thread_x,As[3*block_dim_x*block_dim_y + k*block_dim_y+thread_x],_c_n0,_c_n1,_c_m0,_c_m1);
            }
            __syncthreads();
            //printf("block_x %d block_y %d thread_x %d thread_y %d I %d J %d c_tile %d _c_n0 %f _c_n1 %f \n",block_x,block_y,thread_x,thread_y,I,J,c_tile, _c_n0,_c_n1);
        }
        //if((block_x == (grid_dim_x-1)) && (block_y == (grid_dim_y-1))){
            C[I * N + J] = _c_n0;
            if((I < N) && (J+(npad/2)) < N)
                C[I * N + (J+(npad/2))] = _c_n1;
            if(((I+(npad/2)) < N) && (J < N))
                C[(I + (npad/2)) * N + J] = _c_m0;
            if(((I+(npad/2)) < N) && ((J+(npad/2)) < N))
                C[(I + (npad/2)) * N + (J+(npad/2))] = _c_m1;
            //printf("Result \n");
            //printf("Result block_x %d thread_x %d block_y %d thread_y %d C[%d,%d] %f %f C[%d,%d] %f %f C[%d,%d] %f %f C[%d,%d] %f %f\n",block_x,thread_x,block_y,thread_y, I, J,C[I * N + J],_c_n0,I, (J+(N/2)),C[I * N + (J+(npad/2))],_c_n1,(I + (npad/2)), J,C[(I + (npad/2)) * N + J],_c_m0,(I + (npad/2)),(J+(npad/2)),C[(I + (npad/2)) * N + (J+(npad/2))],_c_m1);
        /*}
        else {
            C[I * N + J] = _c_n0;
            C[I * N + (J+(npad/2))] = _c_n1;
            C[(I + (npad/2)) * N + J] = _c_m0;
            C[(I + (npad/2)) * N + (J+(npad/2))] = _c_m1;
            //printf("Result block_x %d thread_x %d block_y %d thread_y %d C[%d,%d] %f %f C[%d,%d] %f %f C[%d,%d] %f %f C[%d,%d] %f %f\n",block_x,thread_x,block_y,thread_y, I, J,C[I * N + J],_c_n0,I, (J+(N/2)),C[I * N + (J+(npad/2))],_c_n1,(I + (npad/2)), J,C[(I + (npad/2)) * N + J],_c_m0,(I + (npad/2)),(J+(npad/2)),C[(I + (npad/2)) * N + (J+(npad/2))],_c_m1);
        }
    //printf("\n");
*/