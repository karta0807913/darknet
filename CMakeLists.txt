cmake_minimum_required(VERSION 3.5.0)

# find_package(PkgConfig REQUIRED)
find_package (Threads)
include(FindPkgConfig)


include_directories("./include/")
include_directories("./src/")

option(GPU "darknet cuda option" OFF)
option(CUDNN "darknet cudnn" OFF)
option(OPENCV "" OFF)
option(OPENMP "" OFF)
set(CMAKE_VERBOSE_MAKEFILE on)

if(OPENCV)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(BUILD_FLAGS "${BUILD_FLAGS} -DOPENCV")
    list(APPEND LINK_LIBS ${OpenCV_LIBS})
endif(OPENCV)


if(GPU)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(/usr/local/cuda/lib64)
    list(APPEND LINK_LIBS cuda cudart cublas curand)
    CUDA_INCLUDE_DIRECTORIES(./src/ ./include/)
    set(BUILD_FLAGS "${BUILD_FLAGS} -DGPU")
    if(CUDNN)
        set(BUILD_FLAGS "${BUILD_FLAGS} -DCUDNN")
        list(APPEND LINK_LIBS cudnn)
    endif(CUDNN)
    file(GLOB DARKNET_CUDA ./src/*.cu)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
        -gencode arch=compute_30,code=sm_30
        -gencode arch=compute_35,code=sm_35
        -gencode arch=compute_50,code=[sm_50,compute_50]
        -gencode arch=compute_52,code=[sm_52,compute_52]
        "--compiler-options \"${BUILD_FLAGS}\"")
    foreach (target ${DARKNET_CUDA})
        get_filename_component(target_name ${target} NAME)
        string(REGEX MATCH "^([^\\.]*)\\." dummy ${target_name})
        # message("find cuda target ${CMAKE_MATCH_1}")
        set(match ${CMAKE_MATCH_1})
        CUDA_COMPILE(OUT ${target})
        set(BUILD_OBJS ${BUILD_OBJS} ${OUT})
    endforeach(target ${DARKNET_CUDA})
endif(GPU)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BUILD_FLAGS}")

file(GLOB DARKNET_UTIL ./src/*.c ./src/*.cpp)

foreach(target ${DARKNET_UTIL})
    get_filename_component(target_name ${target} NAME)
    string(REGEX MATCH "^([^\\.]*)\\." dummy ${target_name})
    # message("build object ${CMAKE_MATCH_1}")
    add_library(${CMAKE_MATCH_1}.o OBJECT ${target})
    set_property(TARGET ${CMAKE_MATCH_1}.o PROPERTY COMPILE_FLAGS ${BUILD_FLAGS})
    #target_compile_options(${CMAKE_MATCH_1} ${BUILD_FLAGS})
    # target_link_libraries(${CMAKE_MATCH_1} ${CMAKE_THREAD_LIBS_INIT})
    set(BUILD_OBJS ${BUILD_OBJS} $<TARGET_OBJECTS:${CMAKE_MATCH_1}.o>)
endforeach()

add_library(darknet STATIC ${BUILD_OBJS})
target_link_libraries(darknet ${LINK_LIBS} ${CMAKE_THREAD_LIBS_INIT} m stdc++)
set_property(TARGET darknet PROPERTY COMPILE_FLAGS ${BUILD_FLAGS})

add_executable(test_exec test.cpp)

file(GLOB DARKNET_EXAMPLE ./examples/*.c)
add_executable(darknet_out ${DARKNET_EXAMPLE})
target_link_libraries(darknet_out darknet)
set_property(TARGET darknet_out PROPERTY COMPILE_FLAGS ${BUILD_FLAGS})

# add_executable(
