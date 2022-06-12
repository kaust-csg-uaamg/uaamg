cmake_minimum_required(VERSION 3.18)

find_package(TBB CONFIG REQUIRED COMPONENTS tbb)
find_package(OpenVDB CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(SIMD_UAAMG STATIC simd_viscosity3d.h simd_viscosity3d.cpp volume_fractions.cpp volume_fractions.h util.h simd_vdb_poisson.cpp simd_vdb_poisson.h packed3grids.h  packed3grids.cpp vdb_SIMD_IO.h openvdb_grid_math_op.h levelset_util.h levelset_util.cpp SIMD_UAAMG_Ops.h)

target_link_libraries(SIMD_UAAMG PRIVATE Eigen3::Eigen TBB::tbb singletons)
target_link_libraries(SIMD_UAAMG PUBLIC OpenVDB::openvdb)
target_include_directories(SIMD_UAAMG PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#simd compiler flags
if (WIN32)
target_compile_options(SIMD_UAAMG PRIVATE "/arch:AVX2" "/bigobj")
else()
target_compile_options(SIMD_UAAMG PRIVATE "-march=native")
endif()

