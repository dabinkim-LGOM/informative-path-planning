#ifndef SFC_H
#define SFC_H
/**
 * Generate SFC based on JPC. 
 * **/

// #include <grid_map_ipp/grid_map_ipp.hpp>
#include "grid_map_core/GridMap.hpp"
#include <decomp_util/iterative_decomp.h>
#include <SFC/JPS.h>

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
            grid_map::Index cur_index_;
            grid_map::GridMap belief_map_;
            //Frontier cell is in vector of Indices
            grid_map::Index goal_frontier_; // Frontier cell that we want to generate SFC 
            vec_E<Polyhedron<2>> Corridor_;

        public:
            SFC(grid_map::GridMap& map, grid_map::Index& goal_frontier, grid_map::Index& cur_index): belief_map_(map), goal_frontier_(goal_frontier), cur_index_(cur_index)
            {}

            vec_E<Polyhedron<2>> generate_SFC(std::vector<Eigen::Vector2d>& obs);
            vec_E<Polyhedron<2>> get_corridor()
            {
                return Corridor_;
            }
            std::vector<Eigen::Vector2d> JPS_Path();

            void visualize_SFC(vec_E<Polyhedron<2>>& SFC);
    };
}

#endif