cmake_minimum_required(VERSION 3.15)
project(robot_brain_vla CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =====================================================================
# 工业级编译选项 (严抓内存与性能)
# =====================================================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wpedantic")

# 释放性能怪兽：如果是 Debug 模式，开启地址和内存泄漏检查；如果是 Release，极限优化
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address")
endif()

# =====================================================================
# 1. gRPC & Protobuf 环境声明
# =====================================================================
find_program(PROTOC_EXECUTABLE NAMES protoc)
find_program(GRPC_CPP_PLUGIN_EXECUTABLE NAMES grpc_cpp_plugin)

if(NOT PROTOC_EXECUTABLE OR NOT GRPC_CPP_PLUGIN_EXECUTABLE)
    message(FATAL_ERROR "未找到 protoc 或 grpc_cpp_plugin！")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)
pkg_check_modules(PROTOBUF REQUIRED protobuf)

# =====================================================================
# 2. 现代 C++ 库 & ONNX
# =====================================================================
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")

# =====================================================================
# 3. 核心大杀器：引入 ROS2 Humble
# =====================================================================
# 注意：在执行 cmake 之前，必须在终端执行过 source /opt/ros/humble/setup.bash
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# =====================================================================
# 4. 协议编译 (动态生成 robot_brain.pb.cc)
# =====================================================================
set(PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/protos")
set(PROTO_FILE "${PROTO_DIR}/robot_brain.proto")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${GENERATED_DIR})

add_custom_command(
    OUTPUT "${GENERATED_DIR}/robot_brain.pb.cc"
           "${GENERATED_DIR}/robot_brain.pb.h"
           "${GENERATED_DIR}/robot_brain.grpc.pb.cc"
           "${GENERATED_DIR}/robot_brain.grpc.pb.h"
    COMMAND ${PROTOC_EXECUTABLE}
    ARGS --cpp_out="${GENERATED_DIR}"
         --grpc_out="${GENERATED_DIR}"
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
         -I "${PROTO_DIR}"
         "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
)

# =====================================================================
# 5. 构建目标 (Target Definition)
# =====================================================================
# 先把原本的业务代码注释或删除，我们现在只保底层骨架
add_executable(robot_brain_server 
    src/main.cpp 
    src/core/grpc_server.cpp    
    src/core/task_session.cpp   
    src/core/ros_node.cpp 
    src/business/ai_brain.cpp
    "${GENERATED_DIR}/robot_brain.pb.cc" 
    "${GENERATED_DIR}/robot_brain.grpc.pb.cc"
)

# =====================================================================
# 6. 目标属性配置 (缝合 gRPC 与 ROS2)
# =====================================================================
target_include_directories(robot_brain_server PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" 
    "${GENERATED_DIR}" 
    ${GRPC_INCLUDE_DIRS} 
    ${PROTOBUF_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS} 
    "${ONNXRUNTIME_DIR}/include"
)

install(DIRECTORY models config DESTINATION share/${PROJECT_NAME})
# ROS2 包含路径 (AMENT_TARGET_DEPENDENCIES 是一种极简写法)
ament_target_dependencies(robot_brain_server rclcpp ament_index_cpp sensor_msgs geometry_msgs)

target_link_directories(robot_brain_server PRIVATE
    "${ONNXRUNTIME_DIR}/lib"
)

target_link_libraries(robot_brain_server 
    ${GRPC_LIBRARIES}
    ${PROTOBUF_LIBRARIES}
    ${OpenCV_LIBRARIES} 
    spdlog::spdlog
    yaml-cpp
    onnxruntime
    pthread
)

# 告诉 CMake：编译完 robot_brain_server 后，把它拷贝到 install 目录的特定位置
install(TARGETS robot_brain_server
    DESTINATION lib/${PROJECT_NAME}
)

# =====================================================================
# 7. ROS2 包声明尾缀 (必须放在文件的最后一行！)
# =====================================================================
ament_package()