#ifndef GRIDMAPIPP
#define GRIDMAPIPP

// #include "grid_map_core/GridMap.hpp"
#include "grid_map_core/iterators/iterators.hpp"
#include "grid_map_ipp/ObstacleGridConverter.hpp"
#include "grid_map_ipp/wavefront_frontier_detection.hpp"
// #include "grid_map_ipp/util.hpp"
#include <algorithm>
#include "SFC/SFC.hpp"
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>
#include <unordered_set>

using namespace std;

namespace RayTracer{
    
    struct Pose{
        double x; 
        double y; 
        double yaw;

        Pose(double x1, double y1, double yaw1){ x=x1; y=y1; yaw = yaw1; }

    };
    class Lidar_sensor;
    
    class Raytracer{
        private:            
            grid_map::GridMap gt_map_;
            // grid_map::LineIterator raytracer_;

        public:
            // RayTracer(grid_map::Index &startIndex, grid_map::Index &endIndex)
            // {
            //     // raytracer_(startIndex, endIndex);
            // }
            Raytracer(double map_size_x, double map_size_y, int num_obstacle, std::vector<Eigen::Array4d> obstacles)
            {
                grid_map::ObstacleGridConverter converter(map_size_x, map_size_y, num_obstacle, obstacles);
                gt_map_ = converter.GridMapConverter();
            }
            ~Raytracer() {}
            
            grid_map::GridMap get_grid_map(){ return gt_map_;}
            void set_gt_map(grid_map::Matrix &data);
            void set_raytracer();
            pair<vector<grid_map::Index>, bool> raytracing(Lidar_sensor& sensor, grid_map::Index& startIndex, grid_map::Index& endIndex);
            grid_map::Index get_final();
    };


    //Lidar sensor class is permanent for the robot. It has belief_map which recurrently updated with measurement values. 
    class Lidar_sensor{
        private:
            typedef std::vector<std::vector< std::vector<double> > > Cor_vec;
            typedef std::vector< std::vector<double> > Cor; 
            double range_max_;
            double range_min_;
            double hangle_max_;
            double hangle_min_;
            double angle_resol_;
            double resol_;

            double map_size_x_;
            double map_size_y_;
            grid_map::GridMap belief_map_;
            grid_map::Size map_size_;
            Raytracer raytracer_;
            string layer_;

            double ft_cluster_r_ = 5.0;
            vector<Eigen::Vector2d> selected_fts_; //euc reference frame
            
            std::unordered_set<int> obstacles_; //Occupied points are saved in set, in order to find it during SFC generation 
            Eigen::Vector2d submap_length_; //Submap length for local path optimization 

            std::vector<std::pair<vec_E<Polyhedron<2>>, Eigen::Vector2d> > sfc_ft_pair_; //SFC and Frontier point. (euc ref)      
            Cor_vec sfc_jwp_; 

        public:
            Lidar_sensor(double range_max, double range_min, double hangle_max, double hangle_min, double angle_resol, double map_size_x, double map_size_y, double resol, Raytracer& raytracer)
             : range_max_(range_max), range_min_(range_min), hangle_max_(hangle_max), hangle_min_(hangle_min), angle_resol_(angle_resol), map_size_x_(map_size_x), map_size_y_(map_size_y)
             , resol_(resol), raytracer_(raytracer)
             {
                 belief_map_ = init_belief_map();
                 map_size_ = belief_map_.getSize();
                 obstacles_.clear();
                 submap_length_ << 20.0, 20.0;
             }

            ~Lidar_sensor() {}
            
            grid_map::GridMap init_belief_map();              

            void get_measurement(Pose& cur_pos);//Lidar measurement from current pose. 
            pair<vector<grid_map::Index>, bool> gen_single_ray(grid_map::Position& start_pos, grid_map::Position& end_pos); //Single raycasting
            void update_map(vector<grid_map::Index>& free_vec, vector<grid_map::Index>& index_vec); //
            double inverse_sensor(double cur_val, double meas_val);
            
            double get_occ_value(double x, double y)
            {
                Eigen::Vector2d pos_euc(x,y); 
                Eigen::Vector2d pos_grid = grid_map::euc_to_gridref(pos_euc, map_size_);
                grid_map::Position pos(pos_grid(0), pos_grid(1));
                // pos << x, y;
                grid_map::Index idx;
                belief_map_.getIndex(pos, idx);
                return belief_map_.at("base", idx);
            }

            grid_map::GridMap get_belief_map(){
                return belief_map_;
            }
            grid_map::GridMap get_submap(Eigen::Vector2d& pos, Eigen::Array2d& length){
                bool isSuccess = true;
                // grid_map::GridMap full_map = lidar.get_belief_map();
                grid_map::GridMap map = belief_map_.getSubmap(pos, length, isSuccess);
                return map;
            }
            std::unordered_set<int> get_obstacles(){
                return obstacles_;
            }

            void set_belief_map(grid_map::GridMap& gridmap){
                belief_map_ = gridmap;
            }
            

            //Frontier Detector, Return frontier voxels as position(conventional coordinate); 
            vector<Eigen::Vector2d > frontier_detection(grid_map::Position cur_pos);
            vector<vector<Eigen::Vector2d> >  frontier_clustering(vector<Eigen::Vector2d> frontier_pts);
            //By acquisiton functions of frontier points, Python module selects frontier values to generate SFC. 
            void set_selected_frontier(vector<Eigen::Vector2d>& selected_fts)
            {
                selected_fts_ = selected_fts;
            }

            //Construct SFC based on frontiers
            void construct_SFC(Eigen::Vector2d& pos);
            std::vector<std::pair<vec_E<Polyhedron<2>>, Eigen::Vector2d> > get_SFC(){
                return sfc_ft_pair_;
            }
            void construct_SFC_jwp(Eigen::Vector2d& pos);
            Cor_vec get_SFC_jwp(){
                return sfc_jwp_;
            }

            vector<vector<Eigen::Vector2d> > get_JPS_Path(Eigen::Vector2d& pos);
            std::vector<Eigen::Vector2d> generate_obs_grid(Eigen::Vector2d pos);


    };

}

#endif