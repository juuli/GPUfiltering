#SET(CMAKE_OSX_ARCHITECTURES "i386")
#SET(GEOM_PATH "/Developer/HomeMade/geomLib/")
#SET(GEOM_SOURCES ${GEOM_PATH}/inc/geomPrimitives.cpp
# ${GEOM_PATH}/inc/cameraProto.cpp ${GEOM_PATH}/inc/modelBase.cpp ${GEOM_PATH}/inc/meshTools.cpp)

# Define source and include files

# PORT AUDIO
SET(PA_SOURCES ./pa_devs.c)
SET(SOURCES_ ${PA_SOURCES})
SET(PA_INCLUDES ../include/)
SET(PA_LIBRATY ../lib/)

SET(BOOST_INCLUDE "/usr/local/boost_1_53_0")
SET(FFTW_INCLUDE "/Developer/fftw-3.3.2/api")

SET(EXECUTABLE_OUTPUT_PATH ../bin/)

#ADD INCLUDE & LINK DIRECTORIES
INCLUDE_DIRECTORIES(${PA_INCLUDES} ${BOOST_INCLUDE} ${FFTW_INCLUDE} /pa_class/)

# DEFINE EXECUTABLES
ADD_EXECUTABLE(port_audio_class main.cpp logger.cpp ./pa_class/PortAudioClass.cpp ./signal_block/SignalBlock.cpp)

# LINK TARGETS
TARGET_LINK_LIBRARIES(port_audio_class ${PA_LIBRARY}libportaudio.a  ${PA_LIBRARY}libfftw3.a)


