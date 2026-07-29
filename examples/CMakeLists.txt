find_package(CUDAToolkit REQUIRED)

add_executable(cuda_event_guild cuda_event_guild.cc)
target_link_libraries(cuda_event_guild PRIVATE ${CULIB})