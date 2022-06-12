cmake_minimum_required(VERSION 3.18)

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc tbbmalloc_proxy)
find_package(OpenVDB CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

#the FLIP simulation library
add_library(FLIP_vdb STATIC FLIP_vdb.h FLIP_vdb.cpp sdf_vel_pair.h sdf_vel_pair.cpp vdb_velocity_extrapolator.h vdb_velocity_extrapolator.cpp) 
target_include_directories(FLIP_vdb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#the FLIP simulation executable
add_executable(FLIP_simulator main.cpp animation_utils.h animation_utils.cpp)
target_link_libraries(FLIP_simulator PRIVATE SIMD_UAAMG FLIP_vdb TBB::tbb TBB::tbbmalloc_proxy OpenVDB::openvdb singletons)

target_link_libraries(FLIP_vdb PRIVATE TBB::tbb TBB::tbbmalloc)
target_link_libraries(FLIP_vdb PRIVATE OpenVDB::openvdb Eigen3::Eigen SIMD_UAAMG singletons)

#simd compiler flags
if (WIN32)
target_compile_options(FLIP_vdb PRIVATE "/arch:AVX2" "/bigobj")
target_compile_options(FLIP_simulator PRIVATE "/bigobj")
else()
target_compile_options(FLIP_vdb PRIVATE "-march=native")
endif()

install(TARGETS FLIP_simulator DESTINATION "FLIP_simulator")