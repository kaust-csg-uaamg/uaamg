cmake_minimum_required(VERSION 3.18)
find_package(yaml-cpp CONFIG REQUIRED)

add_library(singletons STATIC Timer.h Timer.cpp SimOptions.h SimOptions.cpp YamlSingleton.h YamlSingleton.cpp)
target_include_directories(singletons PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(singletons PUBLIC yaml-cpp)
