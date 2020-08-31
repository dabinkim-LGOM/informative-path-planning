#ifndef SFC_H
#define SFC_H
/**
 * Generate SFC based on JPC. 
 * **/

// #include <grid_map_ipp/grid_map_ipp.hpp>
#include "grid_map_core/GridMap.hpp"
#include <decomp_util/ellipsoid_decomp.h>
#include <chrono>
#include <SFC/JPS.h>

// typedef std::vector<std::vector<std::vector<double> > > cor_type;
using namespace std; 
namespace Planner
{
    //Get Frontier points from current (belief) grid map 
    class SFC
    {   
        protected:
            typedef std::vector<std::vector<double> > cor_type;
            
        private:
            // RayTracer::Lidar_sensor lidar_; 
            grid_map::Index cur_index_;
            grid_map::GridMap belief_map_;
            grid_map::Size size_;
            double world_x_min;
            double world_x_max;
            double world_y_min;
            double world_y_max;
            double box_xy_res; 

            //Frontier cell is in vector of Indices
            grid_map::Index goal_frontier_; // Frontier Index that we want to generate SFC 
            vec_E<Polyhedron<2>> Corridor_;
            cor_type Corridor_jwp_;
            double margin_ = -0.5;
            double SP_EPSILON = 0.0;

            std::vector<Eigen::Vector2d> obs_grid; //Obstacle vector is given with respect to the grid reference frame.
        public:
            SFC(grid_map::GridMap& map, grid_map::Index& goal_frontier, grid_map::Index& cur_index)
            : belief_map_(map), goal_frontier_(goal_frontier), cur_index_(cur_index)
            {
                size_ = map.getSize();
                world_x_min = (-1.0/2.0)*size_(0,0);
                world_x_max = (1.0/2.0)*size_(0,0);
                world_y_min = (-1.0/2.0)*size_(1,0);
                world_y_max = (1.0/2.0)*size_(1,0);
                box_xy_res = map.getResolution();
                // cout << "X MIN: " << world_x_min << endl; 
                // cout << "X MAX: " << world_x_max << endl; 
                // cout << "Y MIN: " << world_y_min << endl; 
                // cout << "Y MAX: " << world_y_max << endl; 
                
                // cout << "BOX RES" << box_xy_res << endl; 
            }
            ~SFC(){}

            vec_E<Polyhedron<2>> generate_SFC();
            void generate_SFC_jwp();
            
            cor_type get_corridor_jwp(){
                return Corridor_jwp_;
            }

            vec_E<Polyhedron<2>> get_corridor()
            {
                return Corridor_;
            }
            std::vector<Eigen::Vector2d> JPS_Path();

            void visualize_SFC(vec_E<Polyhedron<2>>& SFC);

            std::vector<double> expand_box(std::vector<double> &box, double margin);
            bool updateObsBox(std::vector<Eigen::Vector2d> initTraj);
            bool isObstacleInBox(const std::vector<double> &box, double margin);
            bool isBoxInBoundary(const std::vector<double> &box);
            bool isPointInBox(const grid_map::Position  &point, const std::vector<double> &box);
    };
}

#endif