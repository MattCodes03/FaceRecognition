cmake_minimum_required(VERSION 3.7)

if (DEFINED ENV{VCPKG_ROOT})
  set(VCPKG_CMAKE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
else ()
  set(VCPKG_CMAKE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
endif ()

set(CMAKE_TOOLCHAIN_FILE ${VCPKG_CMAKE}) # n.b.

set(n facerec)
project(${n}) # Setting CMAKE_TOOLCHAIN_FILE must precede "project"

find_package(OpenCV REQUIRED) # This must follow the "project" command

add_executable(${n} ${n}.cpp)
target_compile_features(${n} PRIVATE cxx_std_17) # for C++17 <filesystem>
target_include_directories(${n} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${n} ${OpenCV_LIBRARIES})

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${n})

