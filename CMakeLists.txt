cmake_minimum_required(VERSION 3.0.2)
project(livox_backend)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
SET(CMAKE_BUILD_TYPE Release)


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  # livox_ros_driver
  nav_msgs
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  tf
  visualization_msgs
)

find_package(GTSAM REQUIRED QUIET)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)
find_package(Eigen3 3.3 REQUIRED)

catkin_package(
   INCLUDE_DIRS include
   LIBRARIES ${PROJECT_NAME}
  #  CATKIN_DEPENDS livox_ros_driver
   DEPENDS eigen opencv PCL
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
)

link_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${PCL_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(globalOptimization src/globalOptimization.cpp)
add_dependencies(globalOptimization ${catkin_EXPORTED_TARGETS})
# target_link_libraries(globalOptimization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)
target_link_libraries(globalOptimization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)
