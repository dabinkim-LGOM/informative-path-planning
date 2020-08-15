#include <grid_map_ipp/grid_map_ipp.hpp>
#include <grid_map_ipp/grid_map_sdf.hpp>
#include <Eigen/Dense>
#include <grid_map_ipp/util.hpp>
#include <algorithm>
using namespace std;

namespace RayTracer{
    double dist(Eigen::Vector2d pt1, Eigen::Vector2d pt2){
        return sqrt( (pt1(0,0)-pt2(0,0)) * (pt1(0,0)-pt2(0,0)) + (pt1(1,0)-pt2(1,0)) * (pt1(1,0)-pt2(1,0)) );
    }

    grid_map::GridMap Lidar_sensor::init_belief_map()                
    {
        vector<string> name;
        name.clear();
        name.push_back("base");
        vector<string> x = name;
        grid_map::GridMap map(x);

        grid_map::Length len(map_size_x_, map_size_y_);
        grid_map::Position zero(0.0, 0.0); //Zero Position of belief grid
        // zero.x = 0.0; zero.y = 0.0;
        map.setGeometry(len, resol_, zero);
        map.add("base", 0.5); //Initialize map of prob. value with 0.5 (unknown)
        // belief_map_ = map;
        // cout << map.getLayers().at(0) << endl;
        // cout << belief_map_.getLayers().at(0) << endl;

        grid_map::Length size; size = map.getLength();
        // cout << "size " << size(0) << " " << size(1) << endl;
        return map;
    }

    //Transform euclidean (x,y) position value to grid map reference frame
    Eigen::Vector2d Lidar_sensor::euc_to_gridref(Eigen::Vector2d pos)
    {
        // cout << "WHAT?" << endl;
        // pos(0) = pos(0) - map_size_x_/2.0;
        // pos(1) = pos(1) - map_size_y_/2.0;
        // cout << "WHAT2?" << endl;
        //Rotation (-pi/2) w.r.t. z direction
        Eigen::Vector2d grid_pos;
        grid_pos(0) = pos(1) - map_size_x_ /2.0;
        grid_pos(1) = -1.0*pos(0) + map_size_y_ /2.0;
        // cout << "WHAT3?" << grid_pos(0) << " " << grid_pos(1) << endl;
        return grid_pos;
    }
    
    void Lidar_sensor::get_measurement(Pose& cur_pos)
    {   /**
        cur_pos : Eucliden reference frame
        start_pos, end_pos : GridMap frame
        **/

        grid_map::Position pre_transform_pos(cur_pos.x, cur_pos.y);
        grid_map::Position start_pos = euc_to_gridref(pre_transform_pos);
        // cout << start_pos(0) << " " << start_pos(1) << endl;
        grid_map::Index startIndex;
        belief_map_.getIndex(start_pos, startIndex);
        // cout << startIndex(0) << " " << startIndex(1) << endl;

        // cout << "X " << cur_pos.x << " Y " << cur_pos.y << endl;
        vector<grid_map::Index> lidar_free_vec; //Free voxels
        vector<grid_map::Index> lidar_collision_vec; //Occupied voxels
        lidar_free_vec.clear();
        lidar_collision_vec.clear();

        int ray_num = floor( (hangle_max_ - hangle_min_)/angle_resol_ );
        for(int i=0; i< ray_num; i++)
        {   
            double angle = cur_pos.yaw + angle_resol_ * i;
            double end_pos_x = cur_pos.x + range_max_ * cos(angle);
            double end_pos_y = cur_pos.y + range_max_ * sin(angle);

            //Make sure each ray stays in environment range. 
            if(end_pos_x <0.0){
                end_pos_x = 0.1;
            }
            if(end_pos_x >= map_size_x_){
                end_pos_x = map_size_x_ - 0.5;
            }
            if(end_pos_y < 0.0){
                end_pos_y = 0.1;
            }
            if(end_pos_y >= map_size_y_){
                end_pos_y = map_size_y_ - 0.5;
            }

            grid_map::Position pre_transform_pos(end_pos_x, end_pos_y);
            grid_map::Position end_pos(euc_to_gridref(pre_transform_pos));
            pair< vector<grid_map::Index>, bool> idx = gen_single_ray(start_pos, end_pos); //Return free voxel index & true: Occupied voxel
                                                                                           //                          false: no Occupied voxel

            if(idx.second){
                lidar_free_vec.insert(lidar_free_vec.end(), idx.first.begin(), --idx.first.end()); //Concatenate two vectors
                lidar_collision_vec.push_back(idx.first.back());
            }
            else{
                lidar_free_vec.insert(lidar_free_vec.end(), idx.first.begin(), idx.first.end()); //Concatenate two vectors
            }
            // cout <<"After raycasting " << (*(--idx.first.end()))(0) <<" " << (*(--idx.first.end()))(1) <<endl;
            // cout << idx.second << endl;
        }     
        update_map(lidar_free_vec, lidar_collision_vec);
    }


    void Lidar_sensor::update_map(vector<grid_map::Index>& free_vec, vector<grid_map::Index>& occupied_vec)
    {        
        double free = 0.1; double occupied = 0.9;
        double cur_occ_val; double update_occ_val;
        
        //Inverse sensor model
        //1. Free voxels
        for(vector<grid_map::Index>::iterator iter = free_vec.begin(); iter!=free_vec.end(); iter++)
        {
            cur_occ_val = belief_map_.at("base", *iter);
            update_occ_val = inverse_sensor(cur_occ_val, free);
            
            // cout << "CUR " << cur_occ_val << endl;
            // cout << "Free " << update_occ_val << endl;
            belief_map_.at("base", *iter) = update_occ_val;
        }
        //2. Occupied voxels
        for(vector<grid_map::Index>::iterator iter = occupied_vec.begin(); iter!=occupied_vec.end(); iter++)
        {
            cur_occ_val = belief_map_.at("base", *iter);
            update_occ_val = inverse_sensor(cur_occ_val, occupied);
            belief_map_.at("base", *iter) = update_occ_val;
            // cout << "Occ " << update_occ_val << endl;
        } 
    }

    double Lidar_sensor::inverse_sensor(double cur_val, double meas_val)
    {
        double log_cur = log(cur_val / (1.0 - cur_val));
        double log_prior = log( 0.5/ 0.5); 
        double log_meas = log(meas_val / (1.0 - meas_val));

        double log_update = log_meas + log_cur - log_prior;

        return 1.0-1.0/(1.0+exp(log_update));
    }

    pair<vector<grid_map::Index>, bool> Lidar_sensor::gen_single_ray(grid_map::Position& start_pos, grid_map::Position& end_pos) //Single raycasting
    {   
        grid_map::Index startIndex;
        belief_map_.getIndex(start_pos, startIndex);
        grid_map::Index endIndex;
        belief_map_.getIndex(end_pos, endIndex);
        
        // cout << "start_index " <<startIndex(0) << " " << startIndex(1) << endl;
        // cout << startIndex << endl;
        // cout <<"end_pos " << end_pos(0) << " " << end_pos(1) << endl;
        // cout << "end_index " << endIndex(0) << " " << endIndex(1) << endl;
                
        // RayTracer raytracer; 
        pair<vector<grid_map::Index>, bool> result = raytracer_.raytracing(*this, startIndex, endIndex);
        return result;
    }

    /**
     * @brief RayTracing and return pair of voxel indices & whether collision occured. If 2nd element is true, beam is collided & 
     *        the last element of vector is occupied voxel. 
     * 
     * @param sensor 
     * @param startIndex 
     * @param endIndex 
     */
    pair<vector<grid_map::Index>, bool> RayTracer::Raytracer::raytracing(Lidar_sensor& sensor, grid_map::Index& startIndex, grid_map::Index& endIndex)
    {
        grid_map::LineIterator line_iter(gt_map_, startIndex, endIndex);
        vector<grid_map::Index> free_voxel;
        free_voxel.clear();
        grid_map::Index occupied_idx;
        // cout << "Start" << endl;
        for(line_iter; !line_iter.isPastEnd(); ++line_iter)
        {
            //    cout << "Before IF" << endl;
            if(gt_map_.at("base", *line_iter) > 0.95) //Occupied Voxels
            {   //Out of map bound/???
                // cout << "In IF " << *line_iter <<endl;
                occupied_idx = *line_iter;
                free_voxel.push_back(occupied_idx);
                pair<vector<grid_map::Index>, bool> return_pair = make_pair(free_voxel, true);
                return return_pair;
            }
            else
            {
                // cout << "In Else" << endl;
                free_voxel.push_back(*line_iter);
            }
        }
        // cout << "After for loop" << endl;
        pair<vector<grid_map::Index>, bool> return_pair = make_pair(free_voxel, false);

        return return_pair;
    }


    std::vector<Eigen::Vector2d > Lidar_sensor::frontier_detection(grid_map::Position cur_pos){
        grid_map::Frontier ft;
        grid_map::Index cur_idx;
        //Convert conventional xy to grid_map xy coordinate
        grid_map::Size size; 
        size = belief_map_.getSize();
        int x_size = size(1); int y_size = size(0);
        
        grid_map::Position pos_grid;
        pos_grid(0) = cur_pos(1) - y_size/2.0;
        pos_grid(1) = x_size / 2.0 - cur_pos(0);
        
        belief_map_.getIndex(pos_grid, cur_idx);
        cout << "BEFore BEFORE" << endl; 
        cout << cur_idx << endl; 
        vector<vector<grid_map::Index> > frontier_vector = ft.wfd(belief_map_, cur_idx);
        cout << frontier_vector.size() << endl; 
        cout << "BEFore for" << endl; 
        //Convert index from grid_map to conventional xy cooridnate
        vector<Eigen::Vector2d> frontier_position; 
        for(int i=0; i<frontier_vector.size(); ++i){
            for(int j=0; j<frontier_vector.at(i).size(); ++j){
                grid_map::Position trans_pos; 
                belief_map_.getPosition(frontier_vector.at(i).at(j), trans_pos);
                Eigen::Vector2d conv_pos; 
                conv_pos(0) = x_size /2.0 - trans_pos(1);
                conv_pos(1) = y_size/2.0 + trans_pos(0);
                frontier_position.push_back(conv_pos);
            }
        }
        cout << frontier_position.size() << endl; 
        return frontier_position;
    }


    std::vector<std::vector<Eigen::Vector2d> > 
    Lidar_sensor::frontier_clustering(std::vector<Eigen::Vector2d> frontier_pts){
        std::vector<std::vector<Eigen::Vector2d> > clustered_frontiers; //Clustered_frontiers with index 
        // Initial guess
        // Eigen::Vector2d center = frontier_pts.front();
        //Sort frontier points w.r.t. position (x first, then y.)
        //Cut frontier points into clustering points. Based on threshold radius. 
        sort(frontier_pts.begin(), frontier_pts.end(), grid_map::compare);
        grid_map::Print_vec(frontier_pts);
        cout << frontier_pts.size() << endl; 
        int Num = 0;
        
        Eigen::Vector2d cur_center = frontier_pts.at(0);
        std::vector<Eigen::Vector2d> cur_cluster; 
        for(int i=0; i< frontier_pts.size(); i++){

            if( dist(cur_center, frontier_pts.at(i)) < ft_cluster_r_){
                Num++;
                cur_center = cur_center*(Num-1.0)/(Num)+(frontier_pts.at(i))/(double)(Num); 
                cur_cluster.push_back(frontier_pts.at(i));
            }
            else{
                clustered_frontiers.push_back(cur_cluster);
                cur_cluster.clear();
                cout << "Center x: " << cur_center(0,0) << " y: " << cur_center(1,0) << " distance: " << dist(cur_center, frontier_pts.at(i)) << endl; 
                Num = 1;
                cur_center = frontier_pts.at(i);
                cur_cluster.push_back(cur_center);
            }
        }
        return clustered_frontiers;
    }

}