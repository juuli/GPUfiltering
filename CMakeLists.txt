CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
#The name of the project
PROJECT(port_audio_class)
FIND_PACKAGE(CUDA REQUIRED)
find_package(Boost COMPONENTS system filesystem unit_test_framework REQUIRED)

ADD_SUBDIRECTORY(src)

