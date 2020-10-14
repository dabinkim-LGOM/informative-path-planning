#ifndef FFD_HPP
#define FFD_HPP

#include "ros/ros.h"
#include <cstdlib> // Needed for rand()
#include <ctime>
#include <queue>
#include <vector>
#include <map>
#include <algorithm>
// #include "nav_msgs/OccupancyGrid.h"
#include "grid_map_core/GridMap.hpp"


using namespace std;

namespace grid_map{

    struct Point{
        grid_map::Index idx;
        bool visited = false; 
    };

    struct Line{
    std::vector<grid_map::Index> points;
    };


    class Ft_Detector{
        protected:
        typedef std::vector<grid_map::Index> Frontier;

        private:
        std::vector<Frontier> frontiersDB;
        std::vector<grid_map::Index> frontiers;
        std::vector<grid_map::Index> contour;
        std::vector<grid_map::Index> sorted;
        

        // int num_merge = 0; 

        public:
        Ft_Detector(){}
        std::vector<grid_map::Index> FFD( grid_map::Position pose, std::vector<grid_map::Index> lr_idx, const grid_map::GridMap& map);
        std::vector<grid_map::Index> Sort_Polar( std::vector<grid_map::Index> lr_idx, grid_map::Index pose_idx);
        Line Get_Line( grid_map::Index prev, grid_map::Index curr );

        bool is_in_map(grid_map::Size map_size, grid_map::Index cur_index);
        void get_neighbours(grid_map::Index n_array[], grid_map::Index position);
        bool is_frontier_point(const grid_map::GridMap& map, grid_map::Index point);
        
        void update_frontier(grid_map::Position pose, std::vector<grid_map::Index> lr, const grid_map::GridMap& map){
            frontiers = FFD(pose, lr, map);
        }

        //Transform type of frontier set 
        vector<grid_map::Index> get_frontier(){
            return frontiers;
        }
        vector<grid_map::Index> get_contour(){
            return contour;
        }
        vector<grid_map::Index> get_sorted(){
            return sorted;
        }
    };
}


#endif