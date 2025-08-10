cmake_minimum_required(VERSION 3.10)
project(FlowDetector)

# Настройка стандарта C++ и опций компиляции
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra)

# Поиск OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Основной исполняемый файл
add_executable(FlowDetector
    src/main.cpp
    src/FlowDetector.cpp
    src/YOLOv11ONNXDetector.cpp
)
target_link_libraries(FlowDetector ${OpenCV_LIBS})