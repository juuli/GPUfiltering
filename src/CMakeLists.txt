
# PORT AUDIO
SET(PA_SOURCES ./pa_devs.c)

#CUDA STUFF
SET(CUDA_PATH "/usr/local/cuda/")
SET(CUDA_INCLUDE ${CUDA_PATH}/include/)
SET(CUDA_LIBRARY ${CUDA_PATH}/lib/)

SET(BOOST_INCLUDE "/opt/local/include/")
SET(FFTW_INCLUDE "/Developer/fftw-3.3.2/api")

SET(EXECUTABLE_OUTPUT_PATH ../bin/)

#ADD INCLUDE & LINK DIRECTORIES
INCLUDE_DIRECTORIES(/pa_class/ /opt/local/include /usr/local/include ${CUDA_INCLUDE})
LINK_DIRECTORIES(/opt/local/lib ${CUDA_LIBRARY})

#CUDA FILES
CUDA_COMPILE(gpu_filter_o ./filter_block/GPUfilter.cu)
CUDA_COMPILE(cuda_utils_o ./filter_block/cudaUtils.cu)

#EXECUTABLES

# main

ADD_EXECUTABLE(run_pa main.cpp logger.cpp ./pa_class/PortAudioClass.cpp 
               ./signal_block/SignalBlock.cpp ./filter_block/FilterBlock.cpp 
               ${gpu_filter_o} ./filter_block/cudaHelpers.cpp)

ADD_LIBRARY(SweepL ./pa_class/PortAudioClass.cpp 
              ./filter_block/FilterBlock.cpp 
              logger.cpp  ./filter_block/cudaHelpers.cpp
              ./signal_block/SignalBlock.cpp ${gpu_filter_o} 
              ${cuda_utils_o})
# TESTS

ADD_EXECUTABLE(FilterBlock_test ./tests/FilterBlockTest.cpp 
              ./filter_block/FilterBlock.cpp ./filter_block/cudaHelpers.cpp
              logger.cpp  ${gpu_filter_o} ${cuda_utils_o})

ADD_EXECUTABLE(GPUfilter_test ./tests/GPUfilterTest.cpp 
              ./filter_block/cudaHelpers.cpp logger.cpp
              ${gpu_filter_o} ${cuda_utils_o})

ADD_EXECUTABLE(PaClass_test ./tests/PaClassTest.cpp 
              ./filter_block/FilterBlock.cpp 
              logger.cpp ./pa_class/PortAudioClass.cpp ./filter_block/cudaHelpers.cpp
              ./signal_block/SignalBlock.cpp ${gpu_filter_o} ${cuda_utils_o})



# LINK TARGETS

TARGET_LINK_LIBRARIES(run_pa libportaudio.a libfftw3.a cudart)

TARGET_LINK_LIBRARIES(FilterBlock_test libportaudio.a libfftw3.a cudart 
                      ${Boost_FILESYSTEM_LIBRARY} 
                      ${Boost_SYSTEM_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

TARGET_LINK_LIBRARIES(GPUfilter_test cudart cufft libfftw3.a
                      ${Boost_FILESYSTEM_LIBRARY} 
                      ${Boost_SYSTEM_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

TARGET_LINK_LIBRARIES(PaClass_test libportaudio.a  cudart cufft libfftw3.a
                      ${Boost_FILESYSTEM_LIBRARY} 
                      ${Boost_SYSTEM_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

TARGET_LINK_LIBRARIES(SweepL libportaudio.a  cudart cufft libfftw3.a
                      ${Boost_FILESYSTEM_LIBRARY} 
                      ${Boost_SYSTEM_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})


