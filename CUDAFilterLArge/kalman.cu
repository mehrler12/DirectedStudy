#include <chrono>
#include <iostream>
#include "cublas_v2.h"
#include "C:\Users\62793\CSC591\DirectedStudy\data\largeMeasurements.hpp"

#define NUM_OF_MEASUREMENTS 100
#define NUM_OF_MEASURES_IN_FILE 2
#define WINDOW_SIZE 20
#define MEASUREMENT_SIZE 1024
#define STATE_TRANS 0
#define CONTROL_MATRIX MEASUREMENT_SIZE*MEASUREMENT_SIZE
#define IDENTITY_MATRIX MEASUREMENT_SIZE*MEASUREMENT_SIZE *2
#define MEASUREMENT_NOISE MEASUREMENT_SIZE*MEASUREMENT_SIZE *3
#define STATE_TRANS_TRANSPOSE MEASUREMENT_SIZE*MEASUREMENT_SIZE *4
#define IDENTITY_MATRIX_TRANSPOSE MEASUREMENT_SIZE*MEASUREMENT_SIZE *5 //I realize this is just the identity again, wanted to include it for size anyway

void printMatrix(float *a, int rows, int cols){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j <cols;j++){
            if(i < 4 && j <4){
                printf("%6.3f  ",a[j * cols + i]);
            }
        }
        if(i < 4 ){
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

void checkCudaErrors(int line){
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error: %s on line %d\n", cudaGetErrorString(error),line);
        exit(-1);
    }
}

void checkCublasError(cublasStatus_t stat,int line){
    if(stat != CUBLAS_STATUS_SUCCESS){
        printf("CUBLAS error: %d on %d\n",stat,line);
        exit(-1);
    }
}

__global__ void elementSubtractBMinusA(float *a, float *b, int rows, int cols){
    int const index = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("%d\n",index);
    if(index < rows*cols){
        
        a[index] = b[index] - a[index];
        //printf("%d : %f\n",index,a[index]);
    }
}

// __global__ void calcMean(float *a,float *b,int rows,int cols,int stackHeight){
//     __shared__ float redData[THREADS_FOR_RED];
    
//     if(threadIdx.x < stackHeight/2){
//         redData[threadIdx.x] = a[((threadIdx.x*2 ) * (rows*cols)) +blockIdx.x] + a[((threadIdx.x*2 + 1)*(rows*cols))+blockIdx.x];
//     }else{
//         redData[threadIdx.x] = 0;
//     }

//     __syncthreads();

//     for(int i = 1; i < blockDim.x; i *= 2){
//         if( threadIdx.x  % (2 *i) == 0){
//             redData[threadIdx.x] += redData[threadIdx.x + i];
//         }
//         __syncthreads();
//     }
//     b[blockIdx.x] = redData[0]/stackHeight;
// }


__global__ void marker(int *i){
    int const index = threadIdx.x + blockIdx.x * blockDim.x;
    i += index;
}

void checkMatrix(float *r, float *result, int size, int rows, int cols){
    cudaDeviceSynchronize();
    cudaMemcpy(result,r,size,cudaMemcpyDeviceToHost);
    printMatrix(result,rows,cols);
    checkCudaErrors(__LINE__);
}

float kalman(float measurements[][MEASUREMENT_SIZE*MEASUREMENT_SIZE],int num_measurements, int measurement_rows, int measurement_columns){
    float *dev_measurement,*dev_result, *dev_process_noise,*dev_invert_result,*dev_process_error,*dev_kalman_gain,*dev_temp;
    float *result;
    int *dev_info;
    float *dev_residual;
    float *dev_batch_consts;
    float *dev_temp2;
    int *dev_pivot;

    int four_by_four_float_array_size = measurement_columns * measurement_rows* sizeof(float);

    //result = (float*) malloc(four_by_four_float_array_size);
    
    cudaMalloc((void **) &dev_batch_consts,four_by_four_float_array_size*6);
    cudaMalloc((void **) &dev_measurement,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_result,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_process_noise,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_invert_result,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_process_error,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_kalman_gain,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_temp,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_info,sizeof(int));
    cudaMalloc((void **) &dev_residual,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_temp2,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_pivot,measurement_rows*sizeof(int));
    cudaMallocHost((void **) &result, four_by_four_float_array_size);
    float *A[] = { dev_temp };
    float** A_d;
    cudaMalloc<float*>(&A_d,sizeof(A));
    cudaMemcpy(A_d,A,sizeof(A),cudaMemcpyHostToDevice);
    checkCudaErrors(__LINE__);

    float *C[] = { dev_invert_result };
    float** C_d;
    cudaMalloc<float*>(&C_d,sizeof(C));
    cudaMemcpy(C_d,C,sizeof(C),cudaMemcpyHostToDevice);
    checkCudaErrors(__LINE__);
   
    checkCudaErrors(__LINE__);

    cudaMemcpyAsync(dev_batch_consts,batched_const_matrices,four_by_four_float_array_size*6,cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_result,measurements[0],four_by_four_float_array_size,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_process_error,&dev_batch_consts[IDENTITY_MATRIX],four_by_four_float_array_size,cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(dev_process_noise,&dev_batch_consts[IDENTITY_MATRIX],four_by_four_float_array_size,cudaMemcpyDeviceToDevice);

    checkCudaErrors(__LINE__);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1;
    const float beta = 0;

    cublasStatus_t stat;

    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStream_t stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);


    auto start_time = std::chrono::system_clock::now();
    for(int j=1;j<num_measurements;j++){
        int i = j % NUM_OF_MEASURES_IN_FILE;
        //printf("Iteration %d\n",i);
        cudaMemcpyAsync(dev_measurement,measurements[i],four_by_four_float_array_size,cudaMemcpyHostToDevice,stream1);
        checkCudaErrors(__LINE__); 

        //predict
        //A*(x-1)
        cublasSetStream(handle,stream2);
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, &dev_batch_consts[STATE_TRANS], measurement_rows, dev_result, measurement_rows, &beta, dev_result, measurement_rows);
        checkCublasError(stat,1);
        //+Buk
        stat = cublasSaxpy(handle, measurement_rows*measurement_columns,&alpha,&dev_batch_consts[CONTROL_MATRIX],1,dev_result,1);
        checkCublasError(stat,2);
        
        cublasSetStream(handle,stream3);
        //A*(p-1)
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha,  &dev_batch_consts[STATE_TRANS], measurement_rows, dev_process_error, measurement_rows, &beta, dev_process_error, measurement_rows);
        checkCublasError(stat,3);
        //*At
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_process_error, measurement_rows,  &dev_batch_consts[STATE_TRANS_TRANSPOSE], measurement_rows, &beta, dev_process_error, measurement_rows);
        checkCublasError(stat,4);
        //+Q
        stat = cublasSaxpy(handle, measurement_rows*measurement_columns,&alpha,dev_process_noise,1,dev_process_error,1);
        checkCublasError(stat,5);
        
        cublasSetStream(handle,stream2);
        //Calculate Residual
        //H*Xp
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha,  &dev_batch_consts[IDENTITY_MATRIX], measurement_rows, dev_result, measurement_rows, &beta, dev_residual, measurement_rows);
        checkCublasError(stat,12);

        cudaStreamSynchronize(stream1);
        //Y-
        elementSubtractBMinusA<<<measurement_rows,measurement_rows,0,stream2>>>(dev_residual,dev_measurement,measurement_rows,measurement_columns);
        checkCudaErrors(__LINE__);


        //update
        //P*Ht
        cublasSetStream(handle,stream3);
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_process_error, measurement_rows,  &dev_batch_consts[IDENTITY_MATRIX_TRANSPOSE], measurement_rows, &beta, dev_kalman_gain, measurement_rows);
        checkCublasError(stat,6);

        cudaStreamSynchronize(stream3);
        cublasSetStream(handle,stream1);
        //H*P*Ht
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha,  &dev_batch_consts[IDENTITY_MATRIX], measurement_rows, dev_kalman_gain, measurement_rows, &beta, dev_temp, measurement_rows);
        checkCublasError(stat,7);
        //+R
        stat = cublasSaxpy(handle, measurement_rows*measurement_columns,&alpha, &dev_batch_consts[MEASUREMENT_NOISE],1,dev_temp,1);
        checkCublasError(stat,__LINE__);

        stat = cublasSgetrfBatched(handle,measurement_rows,A_d,measurement_rows,dev_pivot,dev_info,1);
        checkCublasError(stat,__LINE__);

        stat = cublasSgetriBatched(handle,measurement_rows,A_d,measurement_rows,dev_pivot,C_d,measurement_rows,dev_info,1);
        //K = (P*Ht)/(H*P*Ht+R)
        checkCublasError(stat,__LINE__);
        cudaDeviceSynchronize();

        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_kalman_gain, measurement_rows, dev_invert_result, measurement_rows, &beta, dev_kalman_gain, measurement_rows);
        checkCublasError(stat,11);

        
        cudaStreamSynchronize(stream1);
        //K*Residual
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_kalman_gain, measurement_rows, dev_residual, measurement_rows, &beta, dev_temp, measurement_rows);
        checkCublasError(stat,13);
        //+Xp
        stat = cublasSaxpy(handle, measurement_rows*measurement_columns,&alpha,dev_temp,1,dev_result,1);
        checkCublasError(stat,14);

        cublasSetStream(handle,stream2);
        //K*H
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_kalman_gain, measurement_rows,  &dev_batch_consts[IDENTITY_MATRIX], measurement_rows, &beta, dev_temp2, measurement_rows);
        checkCublasError(stat,15);
        //I-
        elementSubtractBMinusA<<<measurement_rows,measurement_rows,0,stream2>>>(dev_temp2, &dev_batch_consts[IDENTITY_MATRIX],measurement_rows,measurement_columns);
        checkCudaErrors(__LINE__);
        //*P
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_temp2, measurement_rows, dev_process_error, measurement_rows, &beta, dev_process_error, measurement_rows);
        checkCublasError(stat,16);

        //cudaDeviceSynchronize();
        cudaMemcpyAsync(result,dev_result,four_by_four_float_array_size,cudaMemcpyDeviceToHost,stream1);
        checkCudaErrors(__LINE__);
        //printMatrix(result,measurement_rows,measurement_columns);  
    }
    auto end_time = std::chrono::system_clock::now();
    auto elapsed_time = std::chrono::duration_cast< std::chrono::milliseconds >( end_time - start_time ).count()/static_cast<float>(NUM_OF_MEASUREMENTS);
    //std::cout << "average time per measurment: " << elapsed_time<< " ms" << std::endl;

    cudaFree(dev_measurement);
    cudaFree(dev_result);
    cudaFree(dev_process_noise);
    cudaFree(dev_invert_result);
    cudaFree(dev_process_error);
    cudaFree(dev_kalman_gain);
    cudaFree(dev_temp);
    cudaFree(dev_temp2);
    cudaFree(dev_info);
    cudaFree(dev_residual);
    cudaFree(dev_batch_consts);
    cudaFree(dev_pivot);
    cudaFreeHost(result);
    cublasDestroy(handle);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    checkCudaErrors(__LINE__);
    return elapsed_time;
}

int main(){
    float tpm = 0;
    auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < 1000; i++){
        tpm += kalman(measurements,NUM_OF_MEASUREMENTS,MEASUREMENT_SIZE,MEASUREMENT_SIZE);
    }
    auto end_time = std::chrono::system_clock::now();
    auto elapsed_time = std::chrono::duration_cast< std::chrono::milliseconds >( end_time - start_time );
    std::cout << "average time per run: " << elapsed_time.count() / static_cast< float >( 1000)<< " ms" << std::endl;
    std::cout << "average time per measurment: " << tpm/ static_cast< float >( 1000)<< " ms" << std::endl;

}
