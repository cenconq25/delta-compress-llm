set(TARGET llama-weight-skip-bench)
add_executable(${TARGET} weight-skip-bench.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET2 llama-weight-analysis)
add_executable(${TARGET2} real-model-analysis.cpp)
target_link_libraries(${TARGET2} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET2} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} ${TARGET2} RUNTIME)
endif()
