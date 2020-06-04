# GPU Suitability of Adaptive Kalman Filters
`notes to myself for later are in code backticks`
## Introduction
Earth based telescopes often have to contend with conducting observations through atmospheric turbulence. An interesting binary star system may just look like a smudge with distortion from the atmosphere. Until the 90s the most feasible way to combat turbulence from the atmosphere was to put your telescope in space like Hubble or TESS. The sheer cost of putting your telescope in space is massive and there is only so much room in optimal orbits in space. Adaptive optics allows ground based telescopes to compensate for atmospheric distortion and achieve much higher resolutions. This lets us build massive telescopes on Earth that can observe the far reaches of space or potentially even image exoplanets. 

An adaptive optics system has several parts, a Wavefront Sensor (WFS), a control system, a Deformable Mirror (DM), and potentially one or more Laser Guide Stars (LGS). The WFS takes an image of a bright reference star (can be either a Natural Guide Star (NGS) or LGS) and sends data about the wavefront it received to the control system. The control system takes that data and compares it to how the NGS or LGS is supposed to look and figures out what shape the DM needs to be to make the image how it is supposed to look. The DM is a large mirror made up of a bunch of smaller pieces that can move around incredibly quickly, thus changing the shape of the whole mirror. Given the precision required for getting accurate images and fast changing nature of the atmosphere the control system needs to make its calculations incredibly quickly. 

The most time consuming step is taking the data coming from the WFS and reconstructing the waveform from it. A common method to do this is through an algorithm called Matrix Vector Multiply (MVM) which has O(n^2) time complexity with n being the size of the telescope. As we are building larger and larger telescopes (Thirty Meter Telescope, or the aptly named Extremely Large Telescope) this O(n^2) time complexity will no longer do. There are a large number of algorithms out there but one of particular interest is the Kalman Filter, it has O(n log n) time complexity and provides great accuracy in it's wavefront reconstructions. It also has the added benefit of being able to predict a step ahead fairly accurately. The Kalman filter is also well suited for parallelization on a Graphics Processing Unit (GPU).

A downside of using the Kalman filter is that you need good knowledge of the atmospheric conditions for it to work optimally, if the atmospheric conditions change the accuracy of the filter lowers significantly. An interesting proposed extension of the Kalman filter is the Adaptive Kalman Filter. This allows the algorithm to slightly adjust the matrices that represent the processes like wind speed to better compensate for changing conditions.

This extension to the Kalman Filter leads us to the research question that will be addressed by this report. **Does making the Kalman Filter adaptive add complexity that makes it less efficient to parallelize on a GPU?**. `Reading more on this this may be a real easy to answer question, if that ends up being the case i'll change it to "What approach to adaptive Kalman filtering is best suited for parallelization on a GPU?" and test out other approaches then the estimator one` Answering this question is important as it can tell us whether a wavefront reconstruction method based on Adaptive Kalman Filtering can be implemented on a GPU or whether another approach is required. It is also potentially useful in more than just large telescopes, adaptive optics is also used to image human eyes or to correct for atmospheric distortion in lasers shot into space for either satellite communication or removal of space debris. Adaptive Kalman filters are also used outside of adaptive optics in things like target tracking, robotics, or data fusion of sensor data.

## Background
### Kalman Filtering
Kalman Filtering is a process that allows us to accurately predict a value or set of values based of a series of inaccurate measurements. At a high level it works by predicting what the next state of the system will be and then combines it's prediction with the value it measured based on the error levels in both. 

The Kalman Filter works in 2 phases: predict and update. In the prediction phase it *predicts* the next value of the state as well as the new prediction error. For example if we were using the filter to track a car with GPS we would update our prediction based on the laws of kinematics. We also include some noise in this estimation. `double check this` The final part of the prediction step is updating the predicton noise to use later.
The update stage takes the measured value and *combines* it with the prediction. It does this by first calculating the Kalman Gain which is a ratio of how much it should trust the measurement compared to how much it should trust the prediction. This ratio is based off the error in the measurement and the prediction noise. We then use multiply the ratio by the diffrence between the measurement and the prediction and then add that back to the prediction to get our new state. There is also a noise matrix included in this update to repersent the error in the measurement. We then update the prediction noise again.
These steps are then repeated for every time step to filter the incoming measurements and output a more accurate measurement.

`Add equations`
### Adaptive Kalman Filtering
For the Kalman Filter to effectively work we need to know the various errors that may affect our predictions (i.e the wind will affect atmospheric turbulence so we need to know how the wind will act), in real cases we may not always know these values (wind speed can change very rapidly). We can compensate for this by making the filter adaptive and able to change the error values as it goes along. This lets us compensate for rapid changes in wind speed or other changes in the underlying parameters.

Explain where Adaptive Kalman Filtering extends Kalman Filtering, paper uses least squares error wind profiler to estimate wind conditions every few iterations. Some other variations of AKF with diffrent estimators. Will likely focus on the ones using estimators but if there's time this term I'd love to include a variant I found that essentially runs the Kalman Filter with every likely noise matrix (Wind turbulence in this case) and combines them together, feels like a lot of parrallel potential there. Also include what might make them trickier to parallelize (Potential data dependency on constantly changing process matricies, depends on how it its implemented).
### GPUs
Explain important parts of GPUs and how they achieve their parallelization, warps, threads, streaming multiprocessors, etc.

## Related Work
### Kalman Filtering in Adaptive Optics
Overview of papers on kalman filtering of contributions from papers on KF in AO, will include most papers from the bib
### Adaptive Kalman Filtering
Overview of papers on making the kalman filter adaptive, outline various methods people use. (May not be nessecary as much will be covered in the background section)
### Kalman Filters on GPUs
There's not a ton of papers in this area but overview papers on putting Kalman filters on GPUs

