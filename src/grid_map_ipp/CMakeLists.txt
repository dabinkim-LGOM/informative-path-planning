CMAKE_MINIMUM_REQUIRED ( VERSION 2.8.12 )
PROJECT(grid_map_ipp)
set(ROS_BUILD_TYPE Debug)
# add_definitions(-std=c++17)


find_package(Eigen3 REQUIRED)

find_package(grid_map_core REQUIRED)
find_package(grid_map_sdf REQUIRED)

find_package(decomp_util REQUIRED)

find_package(catkin REQUIRED
            geometry_msgs
            visualization_msgs
            nav_msgs 
            grid_map_ros
            grid_map_msgs
            decomp_ros_msgs
            decomp_ros_utils
            )

catkin_package(
  INCLUDE_DIRS  ${Eigen3_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS}
  LIBRARIES  ${catkin_LIBRARIES} ${OCTOMAP_LIBRARIES} ${grid_map_ipp_LIBRARIES}
  CATKIN_DEPENDS roscpp geometry_msgs visualization_msgs grid_map_msgs nav_msgs decomp_ros_msgs decomp_ros_utils
  )

include_directories(include ${Eigen3_INCLUDE_DIRS} ${grid_map_core_INCLUDE_DIRS} ${grid_map_ros_INCLUDE_DIRS}
                   ${grid_map_msgs_INCLUDE_DIRS} 
                   ${grid_map_sdf_INCLUDE_DIRS} 
                   ${grid_map_ipp_INCLUDE_DIRS})

include_directories(${DECOMP_UTIL_INCLUDE_DIRS})
include_directories(${decomp_ros_utils_INCLUDE_DIRS} ${decomp_ros_utils_INCLUDE_DIRS}/../)

# add_executable(test_grid_map  test/test_gt_map.cpp src/ObstacleGridConverter.cpp)

# add_library(frontier src/wavefront_frontier_detection.cpp)
add_library(grid_map_ipp_library src/grid_map_ipp.cpp src/ObstacleGridConverter.cpp src/SFC.cpp src/JPS.cpp src/util.cpp)
target_include_directories(grid_map_ipp_library PUBLIC  ${DECOMP_UTIL_INCLUDE_DIRS})
target_link_libraries(grid_map_ipp_library ${DECOMP_UTIL_LIBRARIES})

# add_executable(test_lidar test/test_lidar.cpp src/grid_map_ipp.cpp src/ObstacleGridConverter.cpp)
# add_executable(test_frontier test/test_frontier.cpp src/grid_map_ipp.cpp src/ObstacleGridConverter.cpp)
add_executable(test_SFC test/test_SFC.cpp src/grid_map_ipp.cpp src/ObstacleGridConverter.cpp src/JPS.cpp src/SFC.cpp src/util.cpp)

# target_include_directories(test_lidar PUBLIC ${Eigen3_INCLUDE_DIRS} ${grid_map_core_INCLUDE_DIRS}
#                                              ${grid_map_ros_INCLUDE_DIRS} ${grid_map_msgs_INCLUDE_DIRS} 
#                                              ${grid_map_sdf_INCLUDE_DIRS}
#                                              )

# target_include_directories(test_frontier PUBLIC ${Eigen3_INCLUDE_DIRS} ${grid_map_core_INCLUDE_DIRS}
#                           ${grid_map_ros_INCLUDE_DIRS} ${grid_map_msgs_INCLUDE_DIRS} 
#                           ${grid_map_sdf_INCLUDE_DIRS}
#                           )
              
target_include_directories(test_SFC PUBLIC ${Eigen3_INCLUDE_DIRS} ${grid_map_core_INCLUDE_DIRS}
                          ${grid_map_ros_INCLUDE_DIRS} ${grid_map_msgs_INCLUDE_DIRS} 
                          ${grid_map_sdf_INCLUDE_DIRS}  ${decomp_ros_utils_INCLUDE_DIRS}
                          )

# target_link_libraries(test_lidar ${Eigen3_LIBRARIES} ${grid_map_core_LIBRARIES} ${grid_map_ros_LIBRARIES}
#                                  ${grid_map_sdf_LIBRARIES}  
#                                  ${grid_map_ipp_LIBRARIES})

# target_link_libraries(test_frontier ${Eigen3_LIBRARIES} ${grid_map_core_LIBRARIES} ${grid_map_ros_LIBRARIES}
#                       ${grid_map_sdf_LIBRARIES}  
#                       ${grid_map_ipp_LIBRARIES})
target_link_libraries(test_SFC ${Eigen3_LIBRARIES} ${grid_map_core_LIBRARIES} ${grid_map_ros_LIBRARIES}
                      ${grid_map_sdf_LIBRARIES}  
                      ${grid_map_ipp_LIBRARIES} ${decomp_ros_utils_LIBRARIES})
                                 

set(PYBIND11_CPP_STANDARD -std=c++11)
find_package(pybind11 PATHS /usr/local/share/cmake/pybind11)
# add_subdirectory(pybind11)

pybind11_add_module(grid_map_ipp_module python/binding.cpp)
target_link_libraries(grid_map_ipp_module PRIVATE ${Eigen3_LIBRARIES} ${grid_map_core_LIBRARIES} ${grid_map_ros_LIBRARIES}
                       ${grid_map_sdf_LIBRARIES}  
                      grid_map_ipp_library)
