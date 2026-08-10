cmake_minimum_required(VERSION 3.15)
project(robot_brain)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ============================================================
# 依赖查找
# ============================================================

# ROS2
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# gRPC + Protobuf
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# ONNX Runtime
find_package(onnxruntime CONFIG QUIET)
if(NOT onnxruntime_FOUND)
  find_library(ONNXRUNTIME_LIB onnxruntime)
  find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h PATHS /usr/local/include/onnxruntime /usr/include/onnxruntime)
  if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE)
    message(FATAL_ERROR "ONNX Runtime not found")
  endif()
endif()

# spdlog
find_package(spdlog REQUIRED)

# yaml-cpp
find_package(yaml-cpp REQUIRED)

# ============================================================
# Protobuf 代码生成
# ============================================================

set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/protos/robot_brain.proto)
set(PROTO_OUT ${CMAKE_CURRENT_BINARY_DIR}/proto)
file(MAKE_DIRECTORY ${PROTO_OUT})

# gRPC 插件路径
get_target_property(GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin LOCATION)

set(GENERATED_PB ${PROTO_OUT}/robot_brain.pb.cc)
set(GENERATED_GRPC ${PROTO_OUT}/robot_brain.grpc.pb.cc)
set(GENERATED_PB_H ${PROTO_OUT}/robot_brain.pb.h)
set(GENERATED_GRPC_H ${PROTO_OUT}/robot_brain.grpc.pb.h)

add_custom_command(
  OUTPUT ${GENERATED_PB} ${GENERATED_GRPC}
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/protos
    --cpp_out=${PROTO_OUT}
    --grpc_out=${PROTO_OUT}
    --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
    ${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
)

# ============================================================
# 核心 Library
# ============================================================

add_library(${PROJECT_NAME}_core STATIC
  src/core/ros_node.cpp
  src/core/grpc_server.cpp
  src/core/task_session.cpp
  src/business/ai_brain.cpp
  src/business/models/policy_model.cpp
  src/business/models/onnx_model_base.cpp
  src/infra/thread_pool.cpp
  ${GENERATED_PB}
  ${GENERATED_GRPC}
)

target_include_directories(${PROJECT_NAME}_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PROTO_OUT}
  ${ONNXRUNTIME_INCLUDE}
)

target_link_libraries(${PROJECT_NAME}_core
  gRPC::grpc++
  gRPC::grpc++_reflection
  protobuf::libprotobuf
  ${ONNXRUNTIME_LIB}
  spdlog::spdlog
  yaml-cpp
)

ament_target_dependencies(${PROJECT_NAME}_core
  rclcpp sensor_msgs std_msgs geometry_msgs
)

# ============================================================
# 可执行文件
# ============================================================

add_executable(robot_brain_node src/main.cpp)
target_link_libraries(robot_brain_node ${PROJECT_NAME}_core)
ament_target_dependencies(robot_brain_node rclcpp ament_index_cpp sensor_msgs std_msgs geometry_msgs)

# ============================================================
# 安装
# ============================================================

install(TARGETS robot_brain_node
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  config.yaml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.onnx"
)

install(PROGRAMS
  tools/mujoco_sim_node.py
  tools/cloud_client.py
  tools/benchmark_client.py
  tools/convert_g1_policy.py
  tools/generate_proto.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
