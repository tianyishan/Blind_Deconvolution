cmake_minimum_required(VERSION 3.5)
project(prida)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(prida ${SOURCE_FILES})
target_link_libraries (prida ${OpenCV_LIBS})