#ifndef SFC_H
#define SFC_H

#include <grid_map_ipp/grid_map_ipp.hpp>
#include "grid_map_core/GridMap.hpp"

namespace grid_map
{
    //Get Frontier points from current (belief) grid map 
    class SFC
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