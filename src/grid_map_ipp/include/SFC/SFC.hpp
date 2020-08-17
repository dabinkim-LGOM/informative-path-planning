#ifndef SFC_H
#define SFC_H
/**
 * Generate SFC based on JPC. 
 * **/

#include <grid_map_ipp/grid_map_ipp.hpp>
#include "grid_map_core/GridMap.hpp"
#include <decomp_util/iterative_decomp.h>

typedef std::vector<std::vector<std::pair<std::vector<double>, double>>> cor_type;

namespace Planner
{
    //Get Frontier points from current (belief) grid map 
    class SFC
    {   
        protected:
            typedef std::vector<std::vector<std::pair<std::vector<double>, double>>> cor_type;
            
        private:
            // RayTracer::Lidar_sensor lidar_; 
            grid_map::GridMap belief_map_;
            //Frontier cell is in vector of Indices
            Eigen::Vector2d goal_frontier_; // Frontier cell that we want to generate SFC 
            cor_type Corridor_;

        public:
            SFC(grid_map::GridMap& map, Eigen::Vector2d& goal_frontier): belief_map_(map), goal_frontier_(goal_frontier)
            {}
            void generate_SFC(std::vector<Eigen::Vector2d>& obs);
            cor_type get_corridor()
            {
                return Corridor_;
            }
    };
}

#endif