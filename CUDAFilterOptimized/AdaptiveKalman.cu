#include <chrono>
#include <iostream>
#include "cublas_v2.h"
#include "C:\Users\62793\CSC591\DirectedStudy\data\measurements.hpp"

#define WINDOW_SIZE 20
#define THREADS_FOR_RED 10
#define STATE_TRANS 0
#define CONTROL_MATRIX 16
#define IDENTITY_MATRIX 32
#define MEASUREMENT_NOISE 48
#define STATE_TRANS_TRANSPOSE 64
#define IDENTITY_MATRIX_TRANSPOSE 80 //I realize this is just the identity again, wanted to include it for size anyway

const float ax1 =  .1 ,ay1 = .1;
const float ax2 = -.1 ,ay2 = .1;
const float ax3 =  .05,ay3 = .05;
const float ax4 =   0 ,ay4 = .1;

const float noise = 30/2, vnoise = 2/2;

float batched_const_matrices[96] = {1,0,0,0
                                   ,0,1,0,0
                                   ,1,0,1,0
                                   ,0,1,0,1,
                                   .5f*ax1, .5f*ay1  , ax1   , ay1
                                   ,.5f*ax2, .5f*ay2  , ax2   , ay2
                                   ,.5f*ax3, .5f*ay3  , ax3   , ay3
                                   ,.5f*ax4, .5f*ay4  , ax4   , ay4,
                                   1,0,0,0
                                   ,0,1,0,0
                                   ,0,0,1,0
                                   ,0,0,0,1,
                                   noise*noise ,0           ,0             ,0,
                                   0           ,noise*noise ,0             ,0,
                                   0           ,0           ,vnoise*vnoise ,0,
                                   0           ,0           ,0             ,vnoise*vnoise,
                                   1,0,1,0
                                   ,0,1,0,1
                                   ,0,0,1,0
                                   ,0,0,0,1,
                                   1,0,0,0
                                   ,0,1,0,0
                                   ,0,0,1,0
                                   ,0,0,0,1,};



void printMatrix(float *a, int rows, int cols){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j <cols;j++){
            printf("% 11.6f ",a[j * cols + i] );
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void checkCudaErrors(){
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
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
    }
}

__global__ void calcMean(float *a,float *b,int rows,int cols,int stackHeight){
    __shared__ float redData[THREADS_FOR_RED +6];
    
    if(threadIdx.x < THREADS_FOR_RED){
        redData[threadIdx.x] = a[((threadIdx.x*2 ) * (rows*cols)) +blockIdx.x] + a[((threadIdx.x*2 + 1)*(rows*cols))+blockIdx.x];
    }else{
        redData[threadIdx.x] = 0;
    }

    __syncthreads();

    for(int i = 1; i < blockDim.x; i *= 2){
        if( threadIdx.x  % (2 *i) == 0){
            redData[threadIdx.x] += redData[threadIdx.x + i];
        }
        __syncthreads();
    }
    b[blockIdx.x] = redData[0]/stackHeight;
}

__global__ void marker(int *i){
    int const index = threadIdx.x + blockIdx.x * blockDim.x;
    i += index;
}

float kalman(float measurements[][16],int num_measurements, int measurement_rows, int measurement_columns){
    float *dev_measurement,*dev_result, *dev_process_noise,*dev_prediction,*dev_process_error,*dev_kalman_gain,*dev_temp;
    float *result;
    int *dev_info;
    float *dev_residual;
    float *dev_innovation_bank;
    float *dev_batch_consts;
    float *dev_kalman_gain_final;
    float *dev_temp2;

    int four_by_four_float_array_size = measurement_columns * measurement_rows* sizeof(float);

    result = (float*) malloc(four_by_four_float_array_size);
    
    cudaMalloc((void **) &dev_batch_consts,four_by_four_float_array_size*6);
    cudaMalloc((void **) &dev_measurement,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_result,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_process_noise,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_prediction,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_process_error,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_kalman_gain,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_temp,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_info,sizeof(int));
    cudaMalloc((void **) &dev_residual,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_innovation_bank,four_by_four_float_array_size*WINDOW_SIZE);
    cudaMalloc((void **) &dev_temp2,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_kalman_gain_final,four_by_four_float_array_size);

    float *A[] = { dev_temp };
    float** A_d;
    cudaMalloc<float*>(&A_d,sizeof(A));
    cudaMemcpy(A_d,A,sizeof(A),cudaMemcpyHostToDevice);
    checkCudaErrors();
   
    checkCudaErrors();

    cudaMemcpyAsync(dev_batch_consts,batched_const_matrices,four_by_four_float_array_size*6,cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_result,measurements[0],four_by_four_float_array_size,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_process_error,&dev_batch_consts[IDENTITY_MATRIX],four_by_four_float_array_size,cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(dev_process_noise,&dev_batch_consts[IDENTITY_MATRIX],four_by_four_float_array_size,cudaMemcpyDeviceToDevice);

    checkCudaErrors();

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
    for(int i=1;i<num_measurements;i++){
        cudaMemcpyAsync(dev_measurement,measurements[i],four_by_four_float_array_size,cudaMemcpyHostToDevice,stream1);
        checkCudaErrors();
        
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

        //Calculate Residual
        //H*Xp
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha,  &dev_batch_consts[IDENTITY_MATRIX], measurement_rows, dev_result, measurement_rows, &beta, dev_residual, measurement_rows);
        checkCublasError(stat,12);

        cudaStreamSynchronize(stream1);
        //Y-
        elementSubtractBMinusA<<<4,4,0,stream2>>>(dev_residual,dev_measurement,measurement_rows,measurement_columns);
        checkCudaErrors();

        //Adaptation
        //Res*Tranpose(res)
        cublasSetStream(handle,stream2);
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_residual, measurement_rows, dev_residual, measurement_rows, &beta, dev_temp2, measurement_rows);
        checkCublasError(stat,17);
        //Move Residual into bank
        int offset = ((i-1)%WINDOW_SIZE) * measurement_rows* measurement_columns;
        cudaMemcpyAsync(dev_innovation_bank + offset,dev_temp2,four_by_four_float_array_size,cudaMemcpyDeviceToDevice,stream2);
        checkCudaErrors();
        
        if(i>=WINDOW_SIZE){
            calcMean<<<16,16,(THREADS_FOR_RED+6)*sizeof(float),stream2>>>(dev_innovation_bank,dev_temp2,measurement_rows,measurement_columns,WINDOW_SIZE);
            checkCudaErrors();

            cublasSetStream(handle,stream2);
            stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_kalman_gain_final, measurement_rows, dev_temp2, measurement_rows, &beta, dev_temp2, measurement_rows);
            checkCublasError(stat,18);

            stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_temp2, measurement_rows, dev_kalman_gain_final, measurement_rows, &beta, dev_process_noise, measurement_rows);
            checkCublasError(stat,19);
        }

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
        checkCublasError(stat,9);
        
        //K = (P*Ht)/(H*P*Ht+R)
        stat = cublasSmatinvBatched(handle,4,A_d,4,A_d,4,dev_info,1);
        checkCublasError(stat,10);

        cudaDeviceSynchronize();

        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_kalman_gain, measurement_rows, A[0], measurement_rows, &beta, dev_kalman_gain_final, measurement_rows);
        checkCublasError(stat,11);

        
        //cudaStreamSynchronize(stream1);
        //K*Residual
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_kalman_gain_final, measurement_rows, dev_residual, measurement_rows, &beta, dev_temp, measurement_rows);
        checkCublasError(stat,13);
        //+Xp
        stat = cublasSaxpy(handle, measurement_rows*measurement_columns,&alpha,dev_temp,1,dev_result,1);
        checkCublasError(stat,14);

        cublasSetStream(handle,stream2);
        //K*H
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_kalman_gain_final, measurement_rows,  &dev_batch_consts[IDENTITY_MATRIX], measurement_rows, &beta, dev_temp2, measurement_rows);
        checkCublasError(stat,15);
        //I-
        elementSubtractBMinusA<<<4,4,0,stream2>>>(dev_temp2, &dev_batch_consts[IDENTITY_MATRIX],measurement_rows,measurement_columns);
        checkCudaErrors();
        //*P
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_temp2, measurement_rows, dev_process_error, measurement_rows, &beta, dev_process_error, measurement_rows);
        checkCublasError(stat,16);

        //cudaDeviceSynchronize();
        cudaMemcpyAsync(result,dev_result,four_by_four_float_array_size,cudaMemcpyDeviceToHost,stream1);

        //printMatrix(result,measurement_rows,measurement_columns);   
    }
    auto end_time = std::chrono::system_clock::now();
    auto elapsed_time = std::chrono::duration_cast< std::chrono::milliseconds >( end_time - start_time ).count()/static_cast<float>(100);
    //std::cout << "average time per measurment: " << elapsed_time<< " ms" << std::endl;

    cudaFree(dev_measurement);
    cudaFree(dev_result);
    cudaFree(dev_process_noise);
    cudaFree(dev_prediction);
    cudaFree(dev_process_error);
    cudaFree(dev_kalman_gain);
    cudaFree(dev_temp);
    cudaFree(dev_info);
    cudaFree(dev_innovation_bank);
    cudaFree(dev_residual);
    free(result);
    cublasDestroy(handle);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    checkCudaErrors();
    return elapsed_time;
}

int main(){
    float tpm = 0;
    auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < 100; i++){
        tpm += kalman(measurements,100,4,4);
    }
    auto end_time = std::chrono::system_clock::now();
    auto elapsed_time = std::chrono::duration_cast< std::chrono::microseconds >( end_time - start_time );
    std::cout << "average time per run: " << elapsed_time.count() / static_cast< float >( 100)<< " us" << std::endl;
    std::cout << "average time per measurment: " << tpm/100.0<< " ms" << std::endl;

}
