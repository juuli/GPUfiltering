#SET(CMAKE_OSX_ARCHITECTURES "i386")
#SET(GEOM_PATH "/Developer/HomeMade/geomLib/")
#SET(GEOM_SOURCES ${GEOM_PATH}/inc/geomPrimitives.cpp
# ${GEOM_PATH}/inc/cameraProto.cpp ${GEOM_PATH}/inc/modelBase.cpp ${GEOM_PATH}/inc/meshTools.cpp)

# Define source and include files

# PORT AUDIO
SET(PA_SOURCES ./pa_devs.c)
SET(SOURCES_ ${PA_SOURCES})

#CUDA STUFF
SET(CUDA_PATH "/usr/local/cuda/")
SET(CUDA_INCLUDE ${CUDA_PATH}/include/)
SET(CUDA_LIBRARY ${CUDA_PATH}/lib/)

SET(EXECUTABLE_OUTPUT_PATH ../bin/)

#ADD INCLUDE & LINK DIRECTORIES
INCLUDE_DIRECTORIES(/pa_class/ /opt/local/include ${CUDA_INCLUDE})
LINK_DIRECTORIES(/opt/local/lib ${CUDA_LIBRARY})


#CUDA FILES
CUDA_COMPILE(gpu_filter_o ./filter_block/GPUfilter.cu)

# DEFINE EXECUTABLES
ADD_EXECUTABLE(port_audio_class main.cpp logger.cpp ./pa_class/PortAudioClass.cpp ./signal_block/SignalBlock.cpp ./filter_block/FilterBlock.cpp ${gpu_filter_o})

# LINK TARGETS
TARGET_LINK_LIBRARIES(port_audio_class libportaudio.a libfftw3.a cudart)


