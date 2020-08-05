#ifndef FRONTIER_H
#define FRONTIER_H

#include <iostream>
#include <queue>
#include <vector>
#include <list>
#include <grid_map_ipp/grid_map_ipp.hpp>
#include "grid_map_core/GridMap.hpp"

using namespace std;
namespace grid_map
{

    struct grid_ft{
        grid_map::Index index_;
        bool is_MapOpen = false;
        bool is_MapClose = false;
        bool is_FrontierOpen = false;
        bool is_FrontierClose = false;
        grid_ft(grid_map::Index index): index_(index){}
    };
    
    //Get Frontier points from current (belief) grid map 
    class Frontier
    {
        private:
            grid_map::GridMap belief_map_;
            //Frontier cell is in vector of Indices
            list<grid_map::Index&> frontier_cells_;

        public:
            Frontier(grid_map::GridMap map): belief_map_(map){}
            void wfd_frontier(grid_map::Index& pose); //Based on Wavefront Frontier Detector method
            void k_clustering();

    };
}

#endif 