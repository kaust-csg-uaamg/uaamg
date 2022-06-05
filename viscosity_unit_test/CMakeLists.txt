find_package(TBB CONFIG REQUIRED COMPONENTS tbbmalloc_proxy)
find_package(OpenVDB CONFIG REQUIRED)

add_executable(viscosity_unit_test  main.cpp) 
if (${TRACY_ENABLE})
target_sources(viscosity_unit_test PRIVATE ../tracy/TracyClient.cpp)
endif()
target_link_libraries(viscosity_unit_test PRIVATE OpenVDB::openvdb TBB::tbbmalloc_proxy SIMD_UAAMG TracyClient)

