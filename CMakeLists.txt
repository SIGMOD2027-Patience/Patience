cmake_minimum_required(VERSION 3.14)
project(adam_patience LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(patience STATIC src/patience.cpp)
target_include_directories(patience PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(patience PRIVATE -Wall -Wextra -Wpedantic)

add_executable(hnsw_patience_demo examples/hnsw_demo.cpp)
target_include_directories(hnsw_patience_demo SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/hnswlib)
target_link_libraries(hnsw_patience_demo PRIVATE patience pthread)
target_compile_options(hnsw_patience_demo PRIVATE -Wall -Wextra -Wpedantic -march=native)

enable_testing()
add_executable(patience_tests tests/test_patience.cpp)
target_link_libraries(patience_tests PRIVATE patience)
target_compile_options(patience_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME patience_tests COMMAND patience_tests)
