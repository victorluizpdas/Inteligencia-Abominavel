cmake_minimum_required(VERSION 3.10)
project(OpenCVExample)

# 1. Configura as propriedades de saída ANTES de linkar as bibliotecas
add_executable(main main.cpp)

set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "C:/Users/Victor/Desktop/Inteligencia Abominavel"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "C:/Users/Victor/Desktop/Inteligencia Abominavel"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "C:/Users/Victor/Desktop/Inteligencia Abominavel"
)

# 2. Forma moderna e recomendada de incluir e linkar o OpenCV manualmente
target_include_directories(main PRIVATE "C:/opencv/build/include")

target_link_directories(main PRIVATE "C:/opencv/build/x64/vc16/lib")

target_link_libraries(main 
    debug opencv_world480d 
    optimized opencv_world480
)
