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

    struct MyPoint{
    int x;
    int y;
    };

    struct Line{
    std::vector<MyPoint> points;
    };


    class Ft_Detector{
        protected:
        typedef std::vector<MyPoint> Frontier;

        private:
        std::vector<Frontier> frontiersDB;
        std::vector<std::vector<grid_map::Index> > frontiers;

        public:
        Ft_Detector(){}
        std::vector<std::vector<grid_map::Index> > FFD( grid_map::Index pose_idx, std::vector<grid_map::Index> lr, const grid_map::GridMap& map);
        std::vector<MyPoint> Sort_Polar( std::vector<MyPoint> lr, MyPoint pose);
        Line Get_Line( MyPoint prev, MyPoint curr );

        bool is_in_map(grid_map::Size map_size, grid_map::Index cur_index);
        void get_neighbours(grid_map::Index n_array[], grid_map::Index position);
        bool is_frontier_point(const grid_map::GridMap& map, grid_map::Index point);
        
        void update_frontier(grid_map::Index pose_idx, std::vector<grid_map::Index> lr, const grid_map::GridMap& map){
            frontiers = FFD(pose_idx, lr, map);
        }

        //Transform type of frontier set 
        vector<vector<grid_map::Index> > get_frontier(){
            return frontiers;
        }

    };
}


#endif