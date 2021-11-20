cmake_minimum_required(VERSION 3.21)
project(orb-example)

find_package(OpenCV REQUIRED)
find_package(benchmark REQUIRED)

add_executable(show show.cpp)
target_link_directories(show PRIVATE ${OpenCV_LIBRARY_DIR})
target_link_libraries(show PRIVATE ${OpenCV_LIBRARIES})

add_executable(benchmark benchmark.cpp)
target_link_directories(benchmark PRIVATE ${OpenCV_LIBRARY_DIR})
target_link_libraries(benchmark PRIVATE ${OpenCV_LIBRARIES} benchmark::benchmark)
