cmake_minimum_required(VERSION 3.22.1)

project(metaTsetlinMachine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Compiler-specific flags and definitions
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -O3 -mavx512f  -Wno-ignored-attributes -Wno-attributes")
endif()

add_subdirectory(src/pcg)
include_directories(src/pcg)
add_subdirectory(src/metaHeuristic)
add_subdirectory(src/tsetlin)
include_directories(src/tsetlin)
include_directories(src/metaHeuristic/rsa)

add_executable(test demo/test.cpp)
add_executable(xor demo/xor.cpp)
add_executable(rsademo demo/rsaDemo.cpp)

target_link_libraries(test pcgLib tmLib)
target_link_libraries(xor pcgLib tmLib )
target_link_libraries(rsademo pcgLib tmLib rsaLib)