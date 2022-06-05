cmake_minimum_required(VERSION 3.18)
set (CMAKE_CXX_STANDARD 17)

project(mgviscosity3d)

option(TRACY_ENABLE "Enable Tracy profiling of the runtime." OFF)

if(${TRACY_ENABLE})
add_compile_definitions(TRACY_ENABLE)
endif()

add_subdirectory(singletons)
add_subdirectory(tracy)
add_subdirectory(SIMD_UAAMG)
add_subdirectory(FLIP_vdb)
add_subdirectory(viscosity_unit_test)
add_subdirectory(CG_benchmarks)


