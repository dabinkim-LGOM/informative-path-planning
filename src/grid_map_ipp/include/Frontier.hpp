#ifndef FRONTIER_H
#define FRONTIER_H

#include <iostream>
#include <queue>
#include <grid_map_ipp/grid_map_ipp.hpp>
#include "grid_map_core/GridMap.hpp"

using namespace std;
namespace grid_map
{
    //Get Frontier points from current (belief) grid map 
    class Frontier
    {
        private:
            grid_map::GridMap belief_map_;
            //Frontier cell is in vector of Indices
            vector<grid_map::Index> frontier_cells_;

        public:
            void wfd_frontier(); //Based on Wavefront Frontier Detector method
            void k_clustering();

    };
}

#endif 