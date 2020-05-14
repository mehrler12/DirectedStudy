# CSC 591 Directed Studies in GPU-Accelerated Adaptive Optics

[© 2020 Matt Ehrler and Sean Chester](./LICENSE)

## Overview

One of the flagship global research projects is the construction of a thirty-metre telescope: one by a US-led consortium and one by an EU-led consortium. A telescope of this size presents magnificent scientific challenges, not the least of which is _real-time adaptive optics_. A mirror of that size is composed of a grid of small mirrors, each driven by independent actuators, and adaptively adjusting each of them to account for environmental factors is critical to minimising measurement error. This need for real-time responsiveness in a large-scale adaptive system is a classic scenario for hardware acceleration. GPGPU computing can drive scalability up and costs down, but achieving high performance with GPUs requires algorithmic redesign to fit the underlying architecture.

This CSC 591 study will investigate the application of advanced CUDA techniques to accelerate algorithms for Adaptive Optics, with a particular emphasis on research novelty. The primary deliverable is a final research report that demonstrates an effective application of CUDA parallelism and profiling techniques.


## Intended Learning Outcomes (ILO's)

By the end of the course, one should be able to:

  * describe in detail the algorithmic challenges of a particular problem in Adaptive Optics
  * develop a GPU algorithm with high throughput
  * apply advanced knowledge of GPUs---such as memory optimisation, thread saturation, warp utilisation, and asynchronicity---to a concrete implementation
  * conduct profiling and research experiments to appraise the effectiveness of a GPU algorithm
  * prepare a high-quality manuscript and open-source repository to showcase the work


## Structure and Approach

This directed study will be conducted over three months from mid-May to mid-August. It will consist of one long-running project culminating in a final term paper to be submitted in the final week. Three interim reports will be used to assess progress.

There will be weekly virtual meetings on Tuesdays from 19-May-2020 until 18-Aug-2020 (excluding two vacation weeks to be determined). These meetings will serve to discuss challenges and upcoming directions for the project. Technical feedback will be provided via review of GitHub pull requests within six days of the PR being opened; it is recommended that a PR be created at least once per week (depending on the nature of the current task). Reports should also be submitted via PR and evaluation will also take place via the GitHub code review function. The issue manager is a convenient way to keep track of open tasks and overall project status.

If reports are submitted prior to the deadline, they are likely to receive preliminary feedback in advance of formal grading. 


## Evaluation

The overall grade breakdown is given in the table below. Each report will be graded holistically according to the [explicit grading standards set by UVic's English department](https://www.uvic.ca/humanities/english/undergraduate/resources/firstyeargrading/index.php). At least two weeks prior to each deadline, a report-specific rubric and submission guidelines will be created in this GitHub repository.

|Deliverable|Weight|Due Date|Description|
|-----------|------|--------|-----------|
|Interim Report 1|20 %|07-06-2020|Initial report with Lit Review|
|Interim Report 2|20 %|07-07-2020|Preliminary report with Methods|
|Interim Report 3|20 %|07-08-2020|Preliminary report with Experiments|
|Term Paper|40 %|21-08-2020|Full report of project|
