CMAKE_MINIMUM_REQUIRED ( VERSION 2.8.12 )
PROJECT(grid_map_ipp)

# add_definitions(-std=c++17)

find_package(catkin REQUIRED
            geometry_msgs
            visualization_msgs
            nav_msgs 
            grid_map_ros
            grid_map_msgs
            )

find_package(Eigen3 REQUIRED)

find_package(grid_map_core REQUIRED)
find_package(grid_map_sdf REQUIRED)

catkin_package(
  INCLUDE_DIRS  ${Eigen3_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS}
  LIBRARIES  ${catkin_LIBRARIES} ${OCTOMAP_LIBRARIES} ${grid_map_ipp_LIBRARIES}
  CATKIN_DEPENDS roscpp geometry_msgs visualization_msgs grid_map_msgs nav_msgs
  )

include_directories(include ${Eigen3_INCLUDE_DIRS} ${grid_map_core_INCLUDE_DIRS} ${grid_map_ros_INCLUDE_DIRS}
                   ${grid_map_msgs_INCLUDE_DIRS} 
                   ${grid_map_sdf_INCLUDE_DIRS} 
                   ${grid_map_ipp_INCLUDE_DIRS})
# add_executable(test_grid_map  test/test_gt_map.cpp src/ObstacleGridConverter.cpp)

# add_library(frontier src/wavefront_frontier_detection.cpp)
add_library(grid_map_ipp_library src/grid_map_ipp.cpp src/ObstacleGridConverter.cpp )
target_include_directories(grid_map_ipp_library PUBLIC ${FRONTIER_INCLUDE_DIRS})
target_link_libraries(grid_map_ipp_library ${FRONTIER_LIBRARIES})

add_executable(test_lidar test/test_lidar.cpp src/grid_map_ipp.cpp src/ObstacleGridConverter.cpp)

target_include_directories(test_lidar PUBLIC ${Eigen3_INCLUDE_DIRS} ${grid_map_core_INCLUDE_DIRS}
                                             ${grid_map_ros_INCLUDE_DIRS} ${grid_map_msgs_INCLUDE_DIRS} 
                                             ${grid_map_sdf_INCLUDE_DIRS}
                                             )

target_link_libraries(test_lidar ${Eigen3_LIBRARIES} ${grid_map_core_LIBRARIES} ${grid_map_ros_LIBRARIES}
                                 ${grid_map_sdf_LIBRARIES}  
                                 ${grid_map_ipp_LIBRARIES})

                                 
set(PYBIND11_CPP_STANDARD -std=c++11)
find_package(pybind11 PATHS /usr/local/share/cmake/pybind11)
# add_subdirectory(pybind11)

pybind11_add_module(grid_map_ipp_module python/binding.cpp)
target_link_libraries(grid_map_ipp_module PRIVATE ${Eigen3_LIBRARIES} ${grid_map_core_LIBRARIES} ${grid_map_ros_LIBRARIES}
                       ${grid_map_sdf_LIBRARIES}  
                      grid_map_ipp_library)
