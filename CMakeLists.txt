cmake_minimum_required(VERSION 3.18)
set (CMAKE_CXX_STANDARD 17)

set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/bin" CACHE PATH "Directory to place the generated executable binary")
set(X_VCPKG_APPLOCAL_DEPS_INSTALL "ON" CACHE BOOL "Allow Vcpkg to copy dependent libraries to the installed directory")
project(mgviscosity3d)

add_subdirectory(singletons)
add_subdirectory(SIMD_UAAMG)
add_subdirectory(FLIP_vdb)
add_subdirectory(CG_benchmarks)


