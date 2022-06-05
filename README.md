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
Clone the repository, as well as a submodule called "Tracy" for runtime profiling with GUI.
```cmd
> git clone --recursive https://github.com/kaust-csg-uaamg/uaamg
```

This project also depends on a few third party libraries not included in this repository.  
OpenVDB  
Eigen3  
Intel TBB  
yaml-cpp  

We rely on vcpkg to manage these libraries.
Clone from the vcpkg repo
```cmd
> git clone https://github.com/microsoft/vcpkg
```
Install vcpkg:
```cmd
> cd vcpkg
> bootstrap-vcpkg.bat
```

Install dependent libraries in x64 mode:
```cmd
> vcpkg.exe install openvdb:x64-windows eigen3:x64-windows tbb:x64-windows yaml-cpp:x64-windows
```

Now we can build the FILP simulator with the help of CMake.
Take windows CMake GUI as an example.
1. Open CMake GUI, choose the source code path by clicking the "Browse Source" on the top right corner. Set it to our repository, e.g.  "D:/MyCodes/uaamg"
2. Create a new folder to hold the visual studio files. For example, "D:/MyCodes/uaamg/build"
3. Click "+ Add Entry" button. Add a cache entry called "CMAKE_TOOLCHAIN_FILE", which points to a file in vcpkg directory. In our example, it is "D:/MyCodes/vcpkg/scripts/buildsystems/vcpkg.cmake"
4. Click "Configure" button on the bottom left corner. Choose Optional platform for generator as "x64". Click. "Finish" button.
5. Click "Generate" to generate the visual studio solution file.

Open the visual studio solution file in "D:/MyCodes/uaamg/build/mgviscosity3d.sln" and build in Release x64 mode.

