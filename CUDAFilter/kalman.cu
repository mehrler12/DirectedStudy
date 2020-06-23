#include <chrono>
#include <iostream>
#include "cublas_v2.h"
#include "C:\Users\62793\CSC591\DirectedStudy\data\measurements.hpp"

const float ax1 = .1 ,ay1 = .1;
const float ax2 = -.1,ay2 = .1;
const float ax3 = .05,ay3 = .05;
const float ax4 = 0  ,ay4 = .1;

const float noise = 30/2, vnoise = 2/2;

float state_trans_matrix[16] = { 1,0,0,0
                                ,0,1,0,0
                                ,1,0,1,0
                                ,0,1,0,1};

float control_matrix[16] = {.5f*ax1, .5f*ay1  , ax1   , ay1
                           ,.5f*ax2, .5f*ay2  , ax2   , ay2
                           ,.5f*ax3, .5f*ay3  , ax3   , ay3
                           ,.5f*ax4, .5f*ay4  , ax4   , ay4}; 

float identity_matrix[16] = {1,0,0,0
                            ,0,1,0,0
                            ,0,0,1,0
                            ,0,0,0,1};

float measurement_noise[16] = {noise*noise ,0           ,0             ,0,
                               0           ,noise*noise ,0             ,0,
                               0           ,0           ,vnoise*vnoise ,0,
                               0           ,0           ,0             ,vnoise*vnoise};



void printMatrix(float *a, int rows, int cols){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j <cols;j++){
            printf("% 8.3f ",a[j * cols + i] );
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

void checkCublasError(cublasStatus_t stat){
    if(stat != CUBLAS_STATUS_SUCCESS){
        printf("CUBLAS error: %d\n",stat);
        exit(-1);
    }
}

__global__ void elementDivide(float *a, float *b, int rows, int cols){
    int const index = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("%d\n",index);
    if(index < rows*cols){
        if(b[index] != 0){
            a[index] /= b[index];
        }
    }
}
__global__ void elementSubtractBMinusA(float *a, float *b, int rows, int cols){
    int const index = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("%d\n",index);
    if(index < rows*cols){
        a[index] = b[index] - a[index];
    }
}

__global__ void zeroOutNonDiag(float *a,int rows, int cols){
    int const index = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("%d\n",index);
    if(index < rows*cols && index %5 != 0 ){
        a[index] = 0;
    }
}

void kalman(float measurements[][16],int num_measurements, int measurement_rows, int measurement_columns){
    float *dev_measurement, *dev_state_trans_matrix, *dev_result, *dev_process_noise, *dev_measurement_noise,*dev_control_matrix,*dev_prediction,*dev_process_error,*dev_identity_matrix,*dev_kalman_gain,*dev_temp;
    float *result, *testing;
    //printMatrix(measurements[0],4,4);

    int four_by_four_float_array_size = measurement_columns * measurement_rows* sizeof(float);

    result = (float*) malloc(four_by_four_float_array_size);
    testing = (float*) malloc(four_by_four_float_array_size);
    
    cudaMalloc((void **) &dev_measurement,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_state_trans_matrix,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_result,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_process_noise,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_measurement_noise,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_control_matrix,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_prediction,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_process_error,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_identity_matrix,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_kalman_gain,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_temp,four_by_four_float_array_size);
    checkCudaErrors();

    cudaMemcpy(dev_measurement,measurements[1],four_by_four_float_array_size,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_state_trans_matrix,state_trans_matrix,four_by_four_float_array_size,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_identity_matrix,identity_matrix,four_by_four_float_array_size,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_process_noise,dev_identity_matrix,four_by_four_float_array_size,cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_measurement_noise,measurement_noise,four_by_four_float_array_size,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_control_matrix,control_matrix,four_by_four_float_array_size,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_process_error,dev_identity_matrix,four_by_four_float_array_size,cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_result,measurements[0],four_by_four_float_array_size,cudaMemcpyHostToDevice);
    checkCudaErrors();

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1;
    const float beta = 0;

    cublasStatus_t stat;

    for(int i=1;i<num_measurements;i++){
        cudaMemcpy(dev_measurement,measurements[i],four_by_four_float_array_size,cudaMemcpyHostToDevice);
        //predict
        //A*(x-1)
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_state_trans_matrix, measurement_rows, dev_result, measurement_rows, &beta, dev_result, measurement_rows);
        checkCublasError(stat);
        //+Buk
        stat = cublasSaxpy(handle, measurement_rows*measurement_columns,&alpha,dev_control_matrix,1,dev_result,1);
        checkCublasError(stat);

        //A*(p-1)
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_state_trans_matrix, measurement_rows, dev_process_error, measurement_rows, &beta, dev_process_error, measurement_rows);
        checkCublasError(stat);
        //*At
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_process_error, measurement_rows, dev_state_trans_matrix, measurement_rows, &beta, dev_process_error, measurement_rows);
        checkCublasError(stat);
        //+Q
        stat = cublasSaxpy(handle, measurement_rows*measurement_columns,&alpha,dev_process_noise,1,dev_process_error,1);
        checkCublasError(stat);

        //update
        //P*Ht
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_process_error, measurement_rows, dev_identity_matrix, measurement_rows, &beta, dev_kalman_gain, measurement_rows);
        checkCublasError(stat);
        //H*P
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_identity_matrix, measurement_rows, dev_process_error, measurement_rows, &beta, dev_temp, measurement_rows);
        checkCublasError(stat);
        //*Ht
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_temp, measurement_rows, dev_identity_matrix, measurement_rows, &beta, dev_temp, measurement_rows);
        checkCublasError(stat);
        //+R
        stat = cublasSaxpy(handle, measurement_rows*measurement_columns,&alpha,dev_measurement_noise,1,dev_temp,1);
        checkCublasError(stat);
        //K = (P*Ht)/(H*P*Ht+R)
        elementDivide<<<4,4>>>(dev_kalman_gain,dev_temp,measurement_rows,measurement_columns);
        checkCudaErrors();
        //For simplification purposes zero out non diagonal kalman gain entries
        zeroOutNonDiag<<<4,4>>>(dev_kalman_gain,measurement_rows,measurement_columns);
        checkCudaErrors();

        
        //H*Xp
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_identity_matrix, measurement_rows, dev_result, measurement_rows, &beta, dev_temp, measurement_rows);
        checkCublasError(stat);
        //Y-
        elementSubtractBMinusA<<<4,4>>>(dev_temp,dev_measurement,measurement_rows,measurement_columns);
        checkCudaErrors();
        //K*
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_kalman_gain, measurement_rows, dev_temp, measurement_rows, &beta, dev_temp, measurement_rows);
        checkCublasError(stat);
        //+Xp
        stat = cublasSaxpy(handle, measurement_rows*measurement_columns,&alpha,dev_temp,1,dev_result,1);
        checkCublasError(stat);

        //K*H
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_kalman_gain, measurement_rows, dev_identity_matrix, measurement_rows, &beta, dev_temp, measurement_rows);
        checkCublasError(stat);
        //I-
        elementSubtractBMinusA<<<4,4>>>(dev_temp,dev_identity_matrix,measurement_rows,measurement_columns);
        checkCudaErrors();
        //*P
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_temp, measurement_rows, dev_process_error, measurement_rows, &beta, dev_process_error, measurement_rows);
        checkCublasError(stat);
        
        cudaMemcpy(result,dev_result,four_by_four_float_array_size,cudaMemcpyDeviceToHost);
        printMatrix(result,measurement_rows,measurement_columns);
    }
    cudaFree(dev_measurement);
    cudaFree(dev_state_trans_matrix);
    cudaFree(dev_result);
    cudaFree(dev_process_noise);
    cudaFree(dev_measurement_noise);
    cudaFree(dev_control_matrix);
    cudaFree(dev_prediction);
    cudaFree(dev_process_error);
    cudaFree(dev_identity_matrix);
    cudaFree(dev_kalman_gain);
    cudaFree(dev_temp);
    checkCudaErrors();
}

int main(){
    auto start_time = std::chrono::system_clock::now();
    kalman(measurements,100,4,4);
}