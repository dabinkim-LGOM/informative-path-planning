#ifndef UTIL_H
#define UTIL_H
#include <Eigen/Dense>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_ipp/grid_map_ipp.hpp>
#include <grid_map_ipp/ObstacleGridConverter.hpp>

namespace grid_map
{
        
    grid_map::Position get_position(int idx, int idy)
    {   
        grid_map::Position pos;
        grid_map::Index index(idx, idy);
        
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