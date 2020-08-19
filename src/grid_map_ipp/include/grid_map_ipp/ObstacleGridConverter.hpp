#ifndef OBSTACLEGRIDCONVERTER
#define OBSTACLEGRIDCONVERTER

#include <Eigen/Core>
#include "grid_map_core/GridMap.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ipp/util.hpp>
#include <nav_msgs/OccupancyGrid.h>
#include <list>
#include <vector>
#include <ros/ros.h>


namespace grid_map
{
    class ObstacleGridConverter
    {   
        private:
            std::vector<Eigen::Array4d> obstacles_;
            int num_obstacle_;
            double map_size_x_;
            double map_size_y_;

        public:
            // ObstacleGridConverter(double map_size_x, double map_size_y, int num_obstacle, std::vector<Eigen::Array4d> obstacles)
            //  : map_size_x_(map_size_x), map_size_y_(map_size_y), num_obstacle_(num_obstacle), obstacles_(obstacles) {}
            ObstacleGridConverter(double map_size_x, double map_size_y, int num_obstacle, std::vector<Eigen::Array4d> obstacles)
             : map_size_x_(map_size_x), map_size_y_(map_size_y), num_obstacle_(num_obstacle), obstacles_(obstacles) {}

            grid_map::GridMap GridMapConverter(); 
            nav_msgs::OccupancyGrid OccupancyGridConverter();
            nav_msgs::OccupancyGrid OccupancyGridConverter(grid_map::GridMap& gd_map); //Convert given grid_map
            // Eigen::Vector2d euc_to_gridref(Eigen::Vector2d pos);
            // Eigen::Vector2d gridref_to_euc(Eigen::Vector2d pos);


    };
}


#endif