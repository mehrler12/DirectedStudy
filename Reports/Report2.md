# Report 2

## Kalman Filter Implementation

A basic non parallel implementation of the Kalman Filter can be found in the Kalman.ipynb Julia notebook. As the main focus of this project is to evaluate the parallel suitability of the Kalman Filter vs the Adaptive Kalman filter we choose a simpler and more generalizable problem than the Adaptive Optics problem outlined in Report 1. This eases implementation and makes proving correctness much simpler. To make sure the work is still generalizable to AO and other problems we make no optimizations based on the contents of the matrices, as matrices that are sparse in this problem may be dense in others. 

The problem we chose is tracking 4 separate moving objects on a 2d plane. This gives us a 4x4 state matrix as shown:
| Object1 | Object2 | Object3 | Object4 |
|---|---|---|---|
| X Position | X Position | X Position | X Position |
| Y Position | Y Position | Y Position | Y Position |
| X Velocity | X Velocity | X Velocity | X Velocity |
| Y Velocity | Y Velocity | Y Velocity | Y Velocity |
We generate a gold set of measurements for each object by calculating the position of each object every second according to the equation:

<a href="https://www.codecogs.com/eqnedit.php?latex=P&space;=&space;vt&space;&plus;&space;1/2at^2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P&space;=&space;vt&space;&plus;&space;1/2at^2" title="P = vt + 1/2at^2" /></a>

Considering this filter is non adaptive we assume a constant known acceleration.

We then add random noise to the data to get our measurement test set. The gold set of data is shown in Figure 1 and the noisy data is shown in Figure 2.

![ConstGold](..\Figures\ConstantAccelGold.svg)
*Figure 1: Gold Set for Kinematic Data*

![ConstNoise](..\Figures\ConstAccelNoise.svg)
*Figure 2: Noisy Set for Kinematic Data*

For the actual testing we iterate through each measurement and apply the Predict and Update steps of the Kalman Filter described in Report 1. The State Transition and Control Matrices are based off the kinematics equation from above (State Transition handles the velocity section and the Control Matrix handles the acceleration). All transformation matrices as well as the process noise covariance matrix are the identity matrix. The measurement noise covariance matrix is calculated based off the variance of the X Position, Y Position, X Velocity, and Y Velocity considering we know the thresholds for the noise added to them. This is a reasonable assumption as most sensors provide their error as a part of their specifications.   

We can see the results of filtering in Figure 3, the original straight lines from Figure 1 are mostly recovered.

![ConstFilter](..\Figures\ConstAccelFiltered.svg)
*Figure 3: Filtered Set for Kinematic Data*

Additionally the difference between the final position predicted by the filter and the true final position is quite small and is provided below:

| Object1  | Object 2 | Object3 | Object4 |
| -------- | -------- | ------- | ------- | 
| -2.42612 |  0.761943 |  0.210214 |  1.36005 | 
|  1.68932 |  -1.71567 | -0.495081 | -1.03665 |
|  0.201361 |  0.1551  |  0.621997 | -0.0246627 |
|  0.232006 |  0.140774 |  0.410157 | -0.042959 |

One final metric we can look at is the value of the residual (difference between the measurement and the prediction) over time, this gives us an idea of how well the filter is predicting working. In Figure 4 we plot the normalized square of the residual according to [1] and can see it's consistently low.

![ConstFilter](..\Figures\ConstantAccelEps.svg)
*Figure 4: Total Residual Error per measurement*

## Cuda Implementation
After showing our approach works we are able to move our implementation to a parallel architecture with CUDA. The data generated from the Julia implementation is saved into a C++ header file to be used in the CUDA implementation. Instead of 2D arrays we use 1D arrays stored in column major format to make use of the CUBLAS library for highly optimized parallel matrix operations. As no optimization has been done yet besides using CUBLAS the implementation is fairly similar to the Julia Implementation. 

Specific profiling will be discussed later but we can show correctness of this implementation by comparing the final predicted state of the CUDA implementation with the final predicted state of the Julia implementation. Since the Kalman filter is an iterative process any errors throughout the process would cause the final predicted state to change. The final predicted state of the CUDA implementation matches the Julia implementation so we can assume it to be correct.





[1] Bar-Shalom, Yaakov, Xiao-Rong Li, and Thiagalingam Kirubarajan. Estimation with Applications to Tracking and Navigation. New York: Wiley, p. 424, 2001.

