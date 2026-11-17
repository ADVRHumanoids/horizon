The first step is to avoid repeated pinocchio computations due to different functions all requiring forward kinematics at the q, v or a level, jacobians, etc; ideally, depending on the reason why functions are being called (bacward pass, line search, etc), we should be able to have all functions running on a node to avoid repeated calls to pinocchio's update functions.

The next step in this project is to make ilqr suitable for realtime and offline collision and self collision avoidance. 

We shall model a robot (take for instance kyon) links as a set of spheres, using soa tools such as FOAM (already installed), and the environment as a signed distance field (borrowing a lot of inspiration from curobov2).

Here is important to exploit parallelization in the massive amount of sphere-sphere distances as well as SDF evaluations; this could be achieved by CPU parallelization (SIMD) and/or GPU (possibly different backends that you will compare) with a preference to vendor agnostic solutions; a separate CUDA backend could be developed if provides significant improvements or if off-the-shelf libraries are available for CUDA that make the implementation easier. 
Simd would be good for maximum cpu utilization, therefore leaving other cores free for other tasks.

The SDF can be generated from a mesh of the environment, and we can use existing libraries to compute the SDF efficiently. We shall also include common primives for simplified env modeling.

Modifications to the function manager could be required to allow for the parallelization of the collision checking and SDF evaluations. This may involve restructuring the function calls to allow for batch processing of multiple collision checks or SDF evaluations at once.

