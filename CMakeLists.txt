cmake_minimum_required (VERSION 3.10)

project(cxx_differential_equations LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-psabi -Wno-deprecated-declarations -Wold-style-cast")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(cxx_differential_equations INTERFACE)
target_include_directories(cxx_differential_equations INTERFACE include)

add_library(ode_test INTERFACE)
target_include_directories(ode_test INTERFACE .)

add_executable(test_ordinary_diff_eq test_ordinary_diff_eq.cpp)
target_link_libraries(test_ordinary_diff_eq cxx_differential_equations ode_test)
add_test(NAME run_test_ordinary_diff_eq COMMAND bash -c "${CMAKE_BINARY_DIR}/bin/test_ordinary_diff_eq > output/test_ordinary_diff_eq.txt")
