#ifndef UTIL_H
#define UTIL_H
#include <Eigen/Dense>
#include <iostream>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_ipp/grid_map_ipp.hpp>
#include <grid_map_ipp/ObstacleGridConverter.hpp>

namespace grid_map
{
        
    void Print_vec(std::vector<Eigen::Vector2d> frontiers)
    {
        for(int i=0; i<frontiers.size(); i++){
            Eigen::Vector2d cur = frontiers.at(i);
            std::cout << "x: " << cur(0,0) << " y: " << cur(1,0) << std::endl; 
        }
    }

    bool compare(Eigen::Vector2d& t1, Eigen::Vector2d& t2){
        if(t1(0,0)>t2(0,0))
            return false;
        else if(t1(0,0) < t2(0,0))
            return true;
        else{
            if(t1(1,0) > t2(1,0))
                return false;    
            else
            {
                return true; 
            }
        }
    }
}


#endif 