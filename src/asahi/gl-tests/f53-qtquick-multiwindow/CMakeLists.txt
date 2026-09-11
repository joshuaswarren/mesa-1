cmake_minimum_required(VERSION 3.21)

project(f53_qtquick_multiwindow LANGUAGES CXX)

find_package(Qt6 6.11 REQUIRED COMPONENTS Core Gui Qml Quick)

qt_standard_project_setup(REQUIRES 6.11)

qt_add_executable(f53-qtquick-multiwindow
    main.cpp
)

target_compile_features(f53-qtquick-multiwindow PRIVATE cxx_std_17)
target_link_libraries(f53-qtquick-multiwindow PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Qml
    Qt6::Quick
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(f53-qtquick-multiwindow PRIVATE
        -Wall -Wextra -Wpedantic -Werror
    )
endif()
