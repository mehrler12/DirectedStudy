#include <chrono>
#include <iostream>
#include "cublas_v2.h"
#include "C:\Users\62793\CSC591\DirectedStudy\data\measurements.hpp"

#define WINDOW_SIZE 20
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
    int const index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < rows*cols){
        for(int i = 0; i < stackHeight;i++ ){
            b[index] += a[index + (i*rows*cols)];
        }
        b[index] /= stackHeight;
    }

}

void kalman(float measurements[][16],int num_measurements, int measurement_rows, int measurement_columns){
    float *dev_measurement, *dev_state_trans_matrix, *dev_result, *dev_process_noise, *dev_measurement_noise,*dev_control_matrix,*dev_prediction,*dev_process_error,*dev_identity_matrix,*dev_kalman_gain,*dev_temp;
    float *result;
    int *dev_info;
    float *dev_kalman_top,*dev_residual;
    float *dev_innovation_bank;

    int four_by_four_float_array_size = measurement_columns * measurement_rows* sizeof(float);

    result = (float*) malloc(four_by_four_float_array_size);
    
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
    cudaMalloc((void **) &dev_info,sizeof(int));
    cudaMalloc((void **) &dev_kalman_top,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_residual,four_by_four_float_array_size);
    cudaMalloc((void **) &dev_innovation_bank,four_by_four_float_array_size*WINDOW_SIZE);
   
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
        checkCudaErrors();
        //predict
        //A*(x-1)
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_state_trans_matrix, measurement_rows, dev_result, measurement_rows, &beta, dev_result, measurement_rows);
        checkCublasError(stat,1);
        //+Buk
        stat = cublasSaxpy(handle, measurement_rows*measurement_columns,&alpha,dev_control_matrix,1,dev_result,1);
        checkCublasError(stat,2);

        //A*(p-1)
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_state_trans_matrix, measurement_rows, dev_process_error, measurement_rows, &beta, dev_process_error, measurement_rows);
        checkCublasError(stat,3);
        //*At
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_process_error, measurement_rows, dev_state_trans_matrix, measurement_rows, &beta, dev_process_error, measurement_rows);
        checkCublasError(stat,4);
        //+Q
        stat = cublasSaxpy(handle, measurement_rows*measurement_columns,&alpha,dev_process_noise,1,dev_process_error,1);
        checkCublasError(stat,5);

        //Calculate Residual
        //H*Xp
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_identity_matrix, measurement_rows, dev_result, measurement_rows, &beta, dev_residual, measurement_rows);
        checkCublasError(stat,12);
        //Y-
        elementSubtractBMinusA<<<4,4>>>(dev_residual,dev_measurement,measurement_rows,measurement_columns);
        checkCudaErrors();

        //Adaptation
        //Res*Tranpose(res)
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_residual, measurement_rows, dev_residual, measurement_rows, &beta, dev_temp, measurement_rows);
        checkCublasError(stat,17);
        //Move Residual into bank
        int offset = ((i-1)%WINDOW_SIZE) * measurement_rows* measurement_columns;
        cudaMemcpy(dev_innovation_bank + offset,dev_temp,four_by_four_float_array_size,cudaMemcpyDeviceToDevice);
        checkCudaErrors();
        
        if(i>=WINDOW_SIZE){
            printf("test\n");
            calcMean<<<4,4>>>(dev_innovation_bank,dev_temp,measurement_rows,measurement_columns,WINDOW_SIZE);
            checkCudaErrors();

            stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_kalman_gain, measurement_rows, dev_temp, measurement_rows, &beta, dev_temp, measurement_rows);
            checkCublasError(stat,18);

            stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_temp, measurement_rows, dev_kalman_gain, measurement_rows, &beta, dev_process_noise, measurement_rows);
            checkCublasError(stat,19);

            cudaMemcpy(result,dev_process_noise,four_by_four_float_array_size,cudaMemcpyDeviceToHost);
            printMatrix(result,measurement_rows,measurement_columns);
        }


        //update
        //P*Ht
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_process_error, measurement_rows, dev_identity_matrix, measurement_rows, &beta, dev_kalman_gain, measurement_rows);
        checkCublasError(stat,6);
        //H*P
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_identity_matrix, measurement_rows, dev_process_error, measurement_rows, &beta, dev_temp, measurement_rows);
        checkCublasError(stat,7);
        //*Ht
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_temp, measurement_rows, dev_identity_matrix, measurement_rows, &beta, dev_temp, measurement_rows);
        checkCublasError(stat,8);
        //+R
        stat = cublasSaxpy(handle, measurement_rows*measurement_columns,&alpha,dev_measurement_noise,1,dev_temp,1);
        checkCublasError(stat,9);
        
        //K = (P*Ht)/(H*P*Ht+R)
        float *A[] = { dev_temp };
        float** A_d;
        cudaMalloc<float*>(&A_d,sizeof(A));
        cudaMemcpy(A_d,A,sizeof(A),cudaMemcpyHostToDevice);
        checkCudaErrors();

        stat = cublasSmatinvBatched(handle,4,A_d,4,A_d,4,dev_info,1);
        checkCublasError(stat,10);
        
        cudaMemcpy(dev_kalman_top,A[0],four_by_four_float_array_size,cudaMemcpyDeviceToDevice);
        checkCudaErrors();

        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_kalman_gain, measurement_rows, dev_kalman_top, measurement_rows, &beta, dev_kalman_gain, measurement_rows);
        checkCublasError(stat,11);

        
        //K*Residual
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_kalman_gain, measurement_rows, dev_residual, measurement_rows, &beta, dev_temp, measurement_rows);
        checkCublasError(stat,13);
        //+Xp
        stat = cublasSaxpy(handle, measurement_rows*measurement_columns,&alpha,dev_temp,1,dev_result,1);
        checkCublasError(stat,14);

        //K*H
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_kalman_gain, measurement_rows, dev_identity_matrix, measurement_rows, &beta, dev_temp, measurement_rows);
        checkCublasError(stat,15);
        //I-
        elementSubtractBMinusA<<<4,4>>>(dev_temp,dev_identity_matrix,measurement_rows,measurement_columns);
        checkCudaErrors();
        //*P
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, measurement_rows, measurement_columns, measurement_columns, &alpha, dev_temp, measurement_rows, dev_process_error, measurement_rows, &beta, dev_process_error, measurement_rows);
        checkCublasError(stat,16);
        
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