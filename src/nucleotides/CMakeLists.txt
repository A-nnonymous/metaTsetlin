cmake_minimum_required(VERSION 3.22.1)
project(metaTsetlinMachine VERSION 0.1.0)


include_directories(../tsetlin)
aux_source_directory(. NUC_LIB_SRCS)
add_library(nucLib ${NUC_LIB_SRCS})