The next step in this project is to make ilqr suitable for realtime and offline collision and self collision avoidance. 

We shall model a robot (take for instance kyon) links as a set of spheres, using soa tools such as FOAM (already installed), and the environment as a signed distance field (borrowing a lot of inspiration from curobov2).

Here is important to exploit parallelization in the massive amount of sphere-sphere distances as well as SDF evaluations; this could be achieved by CPU parallelization (SIMD) and/or GPU (possibly different backends that you will compare) with a preference to vendor agnostic solutions; a separate CUDA backend could be developed if provides significant improvements or if off-the-shelf libraries are available for CUDA that make the implementation easier. 

The SDF can be generated from a mesh of the environment, and we can use existing libraries to compute the SDF efficiently. We shall also include common primives for simplified env modeling.



