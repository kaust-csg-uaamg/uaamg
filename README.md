# A Fast Unsmoothed Aggregation Algebraic Multigrid Framework for the Large-Scale Simulation of Incompressible Flow

This repository contains a FLIP simulator, a Poisson solver, and a variational viscosity equation solver all based on OpenVDB data structure accelerated by SIMD intrinsics.

Please cite our paper if this repository helps you:  
<pre>
@article{Shao:2022:Multigrid,  
  author    = {Han Shao and Libo Huang and Dominik L.~Michels},  
  title     = {A Fast Unsmoothed Aggregation Algebraic Multigrid Framework for the Large-Scale Simulation of Incompressible Flow},  
  journal   = {ACM Transaction on Graphics},  
  year      = {2022},  
  month     = {07},  
  volume    = {41},  
  number    = {4},  
  articleno = {49},  
  doi       = {https://doi.org/10.1145/3528223.3530109},  
  publisher = {ACM},  
  address   = {New York, NY, USA}  
}  
</pre>

# Build instructions:
Clone the repository
```cmd
git clone https://github.com/kaust-csg-uaamg/uaamg
```
This project also depends on a few third party libraries not included in this repository.  
OpenVDB  
Eigen3  
Intel TBB  
yaml-cpp  

We rely on vcpkg to manage these libraries.
Clone from the vcpkg repo
```cmd
git clone https://github.com/microsoft/vcpkg
```
Install vcpkg:
```cmd
cd vcpkg
bootstrap-vcpkg.bat
```

Install dependent libraries in x64 mode:
```cmd
vcpkg.exe install openvdb:x64-windows eigen3:x64-windows tbb:x64-windows yaml-cpp:x64-windows
```

Now we can build the FILP simulator with the help of CMake.
Take windows CMake GUI as an example.
1. Open CMake GUI, choose the source code path by clicking the "Browse Source" on the top right corner. Set it to our repository, e.g.  "D:/MyCodes/uaamg"
2. Create a new folder to hold the visual studio files. For example, "D:/MyCodes/uaamg/build"
3. Click "+ Add Entry" button. Add a cache entry called "CMAKE_TOOLCHAIN_FILE", which points to a file in vcpkg directory. In our example, it is "D:/MyCodes/vcpkg/scripts/buildsystems/vcpkg.cmake"
4. Click "Configure" button on the bottom left corner. Choose Optional platform for generator as "x64". Click. "Finish" button.
5. Click "Generate" to generate the visual studio solution file.

Open the visual studio solution file in "D:/MyCodes/uaamg/build/mgviscosity3d.sln" and build in Release x64 mode.

If successful, there is an "INSTALL" project in the Visual Studio solution. It builds the FLIP simulator as well as benchmark tests and copy them into a newly created "bin" folder in this directory.

# How To Run a FLIP Scene and Benchmarks
## FLIP Scene
All scenes in our publications can be reproduced and rendered by scene files under "Scene_Houdini". It contains many subfolders, one for each scene. Each folder typically contains a Houdini project file used to generate the simulation geometry, and another file for visualization and rendering. Each subfolder contains README.txt for further instructions.

Our FLIP simulator takes a single yaml configuration file. 
Please check these .yaml config files inside each subfolders in "Scene_Houdini".
In a FLIP config file, the following parameters need to be defined:

**output_prefix** ,string, name of the folder that holds the simulation result.  
**time_step**, float, time step size in seconds, typically 1/24 s.  
**simulation_time**, float, total simulation duration in seconds.  
**init_vel_[x,y,z]**, three floats, initial velocity of the liquid when no velocity field is given in the source, typically zero. It is also the boundary flux velocity for the FLIP domain in some truncated water flow scenes.
**FLIP_domain_collision**, bool, whether the simulation bounding box collides with particles in both the advection step and pressure projection step. In the advection step, particles are pushed away from zero isosurface of collision signed distance fucntions.  
**FLIP_domain_solid**, bool, whether the simulation bounding box is treated as solid in the pressure projection step. This is useful to create an in/out flow boundary which does not collide with particles, but provide a correct Neumann boundary condition in the Poisson projection step.  
**FLIP_solid_friction**, float, value between 0 and 1. The face velocity after pressure projection is further blended according to solid face fractions multiplied by this number. The solid velocity portion is this number multiplied by solid face fraction (also 0 to 1).  
**FLIPCFL**, float, CFL number, maximum number of cells allowed for a particle to travel within a substep. This controls the adaptive time substep size.  
**FLIP_min_step_num**, int, minimum number of substeps within a time step.  
**FLIP_max_step_num**, int, maximum number of substeps within a time step.  
**pressure_acc**, float, pressure projection accuracy.  
**viscosity_acc**, float, variational viscosity solver accuracy.  
**FLIPpool-min[x,y,z]**, three floats, minimum corner of the FLIP solver domain in meters.  
**FLIPpool-max[x,y,z]**, three floats, maximum corner of the FLIP solver domain in meters.  
**FLIPdx**, float, edge length of a voxel in meters.  
**enable_viscosity**, bool, enable viscosity solver.  
**viscosity_coef**, float, dynamic viscosity coefficient.  
**solids_vdb**, string, a file name of static solid collision geometry. That file can contain multiple vdbs. If the name of the vdb is "collision", it influence the particles in the advection step, and velocity field in the projection step. If the name of the vdb is "solid", it only participates in the projection step.  
**source_vdb**, string, a file name of static liquid sources as well as initial liqui geometry. "source_vdb" can contain many vdbs. If the name of the vdb is "initial_liquid_sdf", FLIP particles will be generated in the negative region of that vdb before any simulation steps. If the name of the vdb is "source", new particles will be generated to fill the region where "source" vdb is negative and if the voxel contains less than 8 particles. If a Vec3f vdb grid named "vel" follows any SDF vdb, it is treated as the velocity field from which the generated particle samples initial velocity.  
**dynamic_obj_enabled**, bool, whether dynamic collision geometries are enabled.  
**dynamic_obj_folder**, string, a folder path that contains the dynamic collision geometry.  
**dynamic_obj_prefix**, string, the prefix of dynamic collision objects. Currently, only wavefront obj files with closed triangle mesh are supported. The file name must be "prefix#framenumber.obj". For example if the prefix is "solid.", and the simulation output is frame 0 to frame 120, then it is recommended to have "solid.0.obj" all the way to "solid.121.obj". Topology of the mesh must remain unchanged, since the vertex velocity is calculated based on finite difference forward in time and its position in substeps is linearly interpolated based on two frames.


## Linear Equation Solver Benchmark
After installation, the "bin" folder contains a subfolder called "CG_benchmark", which contains the binary file as well as a config file used for the benchmark of Poisson solver, and the variational viscosity equation solver.
Go inside the "CG_benchmarks" folder, open a command window, and use the following command to run the benchmark:
```cmd
CG_benchmarks.exe config.yaml
```

The config file is copied from "CG_benchmarks/scene/config.yaml".
The following parameters controls its behavior.  

**equation_type**, string, "pressure" or "viscosity", a switch between Poisson equation and variational viscosity equation benchmarks.  
**nx**, int, subdivision count along a single direction. Typically the scene has a characteristic length of 1 meter.  
**output_prefix**, string, folder path that shows some output, such as the geometry of the liquid.  
**poisson_scene_type**, string, "sparse" or "compact". Sparse scene consists of many spherical balls randomly distributed in a 1 meter cube. Compact scene is a box of liquid where only the top face is free surface, while other faces are solid boundaries.  
**pressure_solver_method**, int, available values are [0..5]. 0 uses Eigen diagonal preconditioner conjugate gradient (CG). 1 uses Incomplete Cholesky PCG in Eigen. 2 uses the Incomplete Cholesky PCG from Christopher Batty's code. 3 uses unsmoothed aggregation AMG PCG, but use explicitly built sparse matrix in Eigen. 4 is the method in our paper, UAAMG with SIMD. 5 is AMGCL library.  
**pressure_smoother**, int, available values are [0..3]. 0 is damped Jacobi with weight = 6/7. 1 is scheduled relaxed jacobi. 2 is red black Gauss Seidel. 3 is SPAI0.  
**nballs**, int, number of balls in the sparse scene.  Please see the output sdf to see the exact liqui geometry. 
**resizer**, float, further adjust the size of the ball. Please use the output sdf to visually adjust the size.  
**viscosity_solver_method**, int, available values are [0..3]. 0 is Eigen diagonal PCG. 1 is ICPCG from Batty's code. 2 is UAAMG with explict sparse Eigen matrix. 3 is UAAMG with SIMD. 4 is AMGCL.  
**viscosity_smoother**, int, available values are [3]. 0 is damped jacobi. 1 is scheduled relaxed Jacobi with coefficients optimal for poisson. 2 is multi-color Gauss Seidel smoother. 3 is SPAI0. Note these methods are only for the UAAMG with explicit sparse Eigen matrix, and AMGCL (only damped Jacobi and SPAI0 are available). UAAMG SIMD only use multi-color Gauss Seidel.  
**viscosity_coef**, float, dynamic viscosity coefficient.  


