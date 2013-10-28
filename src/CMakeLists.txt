#SET(CMAKE_OSX_ARCHITECTURES "i386")
#SET(GEOM_PATH "/Developer/HomeMade/geomLib/")
#SET(GEOM_SOURCES ${GEOM_PATH}/inc/geomPrimitives.cpp
# ${GEOM_PATH}/inc/cameraProto.cpp ${GEOM_PATH}/inc/modelBase.cpp ${GEOM_PATH}/inc/meshTools.cpp)

# Define source and include files

# PORT AUDIO
SET(PA_SOURCES ./pa_devs.c)
SET(SOURCES_ ${PA_SOURCES})

#SET(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
#SET(CMAKE_CXX_FLAGS "-m32")

#CUDA STUFF
SET(CUDA_PATH "/usr/local/cuda/")
SET(CUDA_INCLUDE ${CUDA_PATH}/include/)
SET(CUDA_LIBRARY ${CUDA_PATH}/lib/)

SET(BOOST_INCLUDE "/opt/local/include/")
SET(FFTW_INCLUDE "/Developer/fftw-3.3.2/api")

SET(EXECUTABLE_OUTPUT_PATH ../bin/)

#ADD INCLUDE & LINK DIRECTORIES
INCLUDE_DIRECTORIES(/pa_class/ /opt/local/include ${CUDA_INCLUDE} ${GTEST_PATH}/include ${GTEST_PATH}/src)
LINK_DIRECTORIES(/opt/local/lib ${CUDA_LIBRARY} ${GTEST_LIB})

#CUDA FILES
CUDA_COMPILE(gpu_filter_o ./filter_block/GPUfilter.cu)
CUDA_COMPILE(cuda_utils_o ./filter_block/cudaUtils.cu)

# DEFINE EXECUTABLES

# Project
ADD_EXECUTABLE(run_pa main.cpp logger.cpp ./pa_class/PortAudioClass.cpp 
               ./signal_block/SignalBlock.cpp ./filter_block/FilterBlock.cpp 
               ${gpu_filter_o} ./filter_block/cudaHelpers.cpp)

# TESTS
ADD_EXECUTABLE(FilterBlock_test ./tests/FilterBlockTest.cpp 
              ./filter_block/FilterBlock.cpp ./filter_block/cudaHelpers.cpp
              logger.cpp  ${gpu_filter_o} ${cuda_utils_o})

ADD_EXECUTABLE(GPUfilter_test ./tests/GPUfilterTest.cpp 
              ./filter_block/cudaHelpers.cpp logger.cpp
              ${gpu_filter_o} ${cuda_utils_o})

# LINK TARGETS
TARGET_LINK_LIBRARIES(run_pa libportaudio.a libfftw3.a cudart)
TARGET_LINK_LIBRARIES(FilterBlock_test libportaudio.a libfftw3.a cudart 
                      ${Boost_FILESYSTEM_LIBRARY} 
                      ${Boost_SYSTEM_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

TARGET_LINK_LIBRARIES(GPUfilter_test cudart cufft libfftw3.a
                      ${Boost_FILESYSTEM_LIBRARY} 
                      ${Boost_SYSTEM_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})



