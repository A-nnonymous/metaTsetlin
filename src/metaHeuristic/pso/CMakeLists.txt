cmake_minimum_required(VERSION 3.22.1)
project(metaTsetlinMachine VERSION 0.1.0)

include_directories(../../pcg)
aux_source_directory(. PSO_LIB_SRCS)
add_library(psoLib ${PSO_LIB_SRCS})