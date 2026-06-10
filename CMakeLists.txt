cmake_minimum_required(VERSION 3.10)

project(OpenCVExample)

include_directories("C:/opencv/build/include")
link_directories("C:/opencv/build/x64/vc16/lib")

add_executable(main main.cpp)

target_link_libraries(main
    debug opencv_world480d
    optimized opencv_world480
)