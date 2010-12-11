

include_directories(${CMAKE_SOURCE_DIR}/aquila/src)
link_directories(${CMAKE_SOURCE_DIR}/aquila/lib)

set(SRCS features.cpp main.cpp layer.cpp neuralnet.cpp)
add_executable(genre ${SRCS})
target_link_libraries(genre aquila boost_filesystem boost_system)
set_target_properties(genre PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

