cmake_minimum_required(VERSION 3.22.1)

project(metaTsetlinMachine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Compiler-specific flags and definitions
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -O3 -g -mavx512f  -Wno-ignored-attributes -Wno-attributes")
endif()

add_subdirectory(src/pcg)
include_directories(src/pcg)
add_subdirectory(src/metaHeuristic)
add_subdirectory(src/tsetlin)
include_directories(src/tsetlin)
include_directories(src/nucleotides)
add_subdirectory(src/nucleotides)
include_directories(src/metaHeuristic/rsa)
include_directories(src/metaHeuristic/pso)
include_directories(src/metaHeuristic/aoa)

add_executable(meta demo/metaTsetlinDemo.cpp)
#add_executable(test demo/test.cpp)
#add_executable(xor demo/xor.cpp)
#add_executable(pso demo/psotest.cpp)
#add_executable(aoa demo/aoatest.cpp)
#add_executable(rsademo demo/rsaDemo.cpp)

#target_link_libraries(test nucLib pcgLib tmLib )
#target_link_libraries(xor pcgLib nucLib tmLib )
#target_link_libraries(pso pcgLib psoLib )
#target_link_libraries(aoa pcgLib aoaLib )
target_link_libraries(meta pcgLib nucLib tmLib rsaLib psoLib aoaLib)