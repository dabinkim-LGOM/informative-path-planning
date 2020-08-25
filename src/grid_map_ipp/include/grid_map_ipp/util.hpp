#ifndef UTIL_H
#define UTIL_H
#include <Eigen/Dense>
#include <iostream>
#include <grid_map_core/GridMap.hpp>
// #include <grid_map_ipp/grid_map_ipp.hpp>
// #include <grid_map_ipp/ObstacleGridConverter.hpp>

namespace grid_map
{
        //Transform euclidean (x,y) position value to grid map reference frame
        Eigen::Vector2d euc_to_gridref(Eigen::Vector2d&, Eigen::Array2i&);

        Eigen::Vector2d grid_to_eucref(Eigen::Vector2d&, Eigen::Array2i&);

        std::vector<double> gridbox_to_eucbox(std::vector<double>&, Eigen::Array2i&);

        void Print_vec(std::vector<Eigen::Vector2d>&);

        bool compare(Eigen::Vector2d&, Eigen::Vector2d&);

}


#endif 