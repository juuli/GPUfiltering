CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
#The name of the project
PROJECT(port_audio_class)
FIND_PACKAGE(CUDA REQUIRED)
find_package(Boost COMPONENTS system filesystem unit_test_framework REQUIRED)

SET (CMAKE_C_COMPILER             "/usr/bin/clang")
SET (CMAKE_C_FLAGS                "-Wall -std=c99")
SET (CMAKE_C_FLAGS_DEBUG          "-g")
SET (CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_C_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")

#SET (CMAKE_CXX_COMPILER             "/usr/local/bin/clang++")
SET (CMAKE_CXX_FLAGS                "-Wall")
#SET (CMAKE_CXX_COMPILER             "/opt/bin/llvm-g++")
SET (CMAKE_CXX_FLAGS                "-Wall")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

ADD_SUBDIRECTORY(src)

