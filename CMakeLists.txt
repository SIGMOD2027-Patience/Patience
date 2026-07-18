cmake_minimum_required(VERSION 3.14)
project(adam_patience LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(patience STATIC src/patience.cpp src/graph_search.cpp)
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

add_executable(graph_matrix_tests tests/test_graph_matrix.cpp)
target_link_libraries(graph_matrix_tests PRIVATE patience)
target_compile_options(graph_matrix_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME graph_matrix_tests COMMAND graph_matrix_tests)

add_executable(all_graphs_metrics examples/all_graphs_metrics.cpp)
target_link_libraries(all_graphs_metrics PRIVATE patience)
target_compile_options(all_graphs_metrics PRIVATE -Wall -Wextra -Wpedantic)

add_executable(mnist_hnsw experiments/mnist_hnsw.cpp)
target_include_directories(mnist_hnsw SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/hnswlib)
target_link_libraries(mnist_hnsw PRIVATE patience pthread)
target_compile_options(mnist_hnsw PRIVATE -Wall -Wextra -Wpedantic -march=native)
