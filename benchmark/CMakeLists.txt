find_package(benchmark REQUIRED)
find_package(Threads)

add_executable(find find.cpp)
target_link_libraries(find benchmark ${CMAKE_THREAD_LIBS_INIT})
