#include <grid_map_ipp/grid_map_ipp.hpp>
#include <grid_map_ipp/grid_map_sdf.hpp>

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
        // name.push_back("JPS");
        vector<string> x = name;
        grid_map::GridMap map(x);

        grid_map::Length len(map_size_x_, map_size_y_);
        grid_map::Position zero(0.0, 0.0); //Zero Position of belief grid
        // zero.x = 0.0; zero.y = 0.0;
        map.setGeometry(len, resol_, zero);
        map.add("base", 0.5); //Initialize map of prob. value with 0.5 (unknown)
        map.add("JPS", 0);
        // belief_map_ = map;
        // cout << map.getLayers().at(0) << endl;
        // cout << belief_map_.getLayers().at(0) << endl;

        grid_map::Length size; size = map.getLength();
        // cout << "size " << size(0) << " " << size(1) << endl;
        return map;
    }

    //Main Loop for Lidar Sensor measurement
    void Lidar_sensor::get_measurement(Pose& cur_pos)
    {   /**
        cur_pos : Eucliden reference frame
        start_pos, end_pos : GridMap frame
        **/
        grid_map::Size map_size = belief_map_.getSize();
        grid_map::Position pre_transform_pos(cur_pos.x, cur_pos.y);
        grid_map::Position start_pos = grid_map::euc_to_gridref(pre_transform_pos, map_size);
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
            grid_map::Position end_pos(grid_map::euc_to_gridref(pre_transform_pos, map_size));
            pair< vector<grid_map::Index>, bool> idx = gen_single_ray(start_pos, end_pos); //Return free voxel index & true: Occupied voxel
                                                                                           //                          false: no Occupied voxel

            if(idx.second){
                lidar_free_vec.insert(lidar_free_vec.end(), idx.first.begin(), --idx.first.end()); //Concatenate two vectors
                lidar_collision_vec.push_back(idx.first.back());
                auto index = idx.first.back();
                int int_idx = map_size_x_*index(0,0) + index(1,0);
                obstacles_.insert(int_idx); //Insert Obstacle Index into unordered_map 
            }
            else{
                lidar_free_vec.insert(lidar_free_vec.end(), idx.first.begin(), idx.first.end()); //Concatenate two vectors
            }
            // cout <<"After raycasting " << (*(--idx.first.end()))(0) <<" " << (*(--idx.first.end()))(1) <<endl;
            // cout << idx.second << endl;
        }     
        update_map(lidar_free_vec, lidar_collision_vec);
        cur_scan_ = lidar_collision_vec;
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

    std::vector<Eigen::Vector2d > Lidar_sensor::FFD(grid_map::Position pos_euc){
        grid_map::Ft_Detector ft;
        grid_map::Index cur_idx;
        //Convert conventional xy to grid_map xy coordinate
        grid_map::Size size; 
        size = belief_map_.getSize();
        int x_size = size(1); int y_size = size(0);
        
        grid_map::Position pos_grid = grid_map::euc_to_gridref(pos_euc, map_size_);
        
        belief_map_.getIndex(pos_grid, cur_idx);
        
        //Update Frontier if there is new scan data 
        if(cur_scan_.size()>0){
            ft.update_frontier(cur_idx,cur_scan_, belief_map_);
            std::cout << "After Update" << std::endl; 
            vector<vector<grid_map::Index> > frontier_vector = ft.get_frontier();
            std::cout << "After Get Frontier" << std::endl; 
            //Convert index from grid_map to conventional xy cooridnate
            vector<Eigen::Vector2d> frontier_position; 
            for(int i=0; i<frontier_vector.size(); ++i){
                for(int j=0; j<frontier_vector.at(i).size(); ++j){
                    grid_map::Position trans_pos; 
                    belief_map_.getPosition(frontier_vector.at(i).at(j), trans_pos);
                    Eigen::Vector2d conv_pos = grid_map::grid_to_eucref(trans_pos, map_size_); 
                    // conv_pos(0) = x_size /2.0 - trans_pos(1);
                    // conv_pos(1) = y_size/2.0 + trans_pos(0);
                    frontier_position.push_back(conv_pos);
                }
            }
            cur_frontier_set_ = frontier_position; 
            // return frontier_position;
        }

        return cur_frontier_set_;
    }


    std::vector<Eigen::Vector2d > Lidar_sensor::frontier_detection(grid_map::Position pos_euc){
        grid_map::Frontier ft;
        grid_map::Index cur_idx;
        //Convert conventional xy to grid_map xy coordinate
        grid_map::Size size; 
        size = belief_map_.getSize();
        int x_size = size(1); int y_size = size(0);
        
        grid_map::Position pos_grid = grid_map::euc_to_gridref(pos_euc, map_size_);
        // pos_grid(0) = pos_euc(1) - y_size/2.0;
        // pos_grid(1) = x_size / 2.0 - pos_euc(0);
        
        belief_map_.getIndex(pos_grid, cur_idx);

        vector<vector<grid_map::Index> > frontier_vector = ft.wfd(belief_map_, cur_idx);

        //Convert index from grid_map to conventional xy cooridnate
        vector<Eigen::Vector2d> frontier_position; 
        for(int i=0; i<frontier_vector.size(); ++i){
            for(int j=0; j<frontier_vector.at(i).size(); ++j){
                grid_map::Position trans_pos; 
                belief_map_.getPosition(frontier_vector.at(i).at(j), trans_pos);
                Eigen::Vector2d conv_pos = grid_map::grid_to_eucref(trans_pos, map_size_); 
                // conv_pos(0) = x_size /2.0 - trans_pos(1);
                // conv_pos(1) = y_size/2.0 + trans_pos(0);
                frontier_position.push_back(conv_pos);
            }
        }
        return frontier_position;
    }

    std::vector<Eigen::Vector2d>
    Lidar_sensor::frontier_kmeans(std::vector<Eigen::Vector2d>& frontier_pts){
        int total_size = frontier_pts.size();
        int cluster_num = std::ceil(total_size / 20.0); 
        int iter = 5;
        clustering::KMeans km(cluster_num, iter);

        std::vector<clustering::Point> points; 
        for(int i=0; i< total_size; i++){
            clustering::Point pt(i, frontier_pts[i]);
            points.push_back(pt);
        }
        km.run(points);
        std::vector<clustering::Cluster> cluster_results;
        cluster_results = km.get_cluster();

        std::vector<Eigen::Vector2d> centroid_vec;
        for(int i=0; i<cluster_results.size(); i++){
            std::vector<double> pos = cluster_results[i].getCentroidByPos();
            Eigen::Vector2d eigen_pos(pos[0], pos[1]);
            centroid_vec.push_back(eigen_pos);
        }
        clustered_fts_ = centroid_vec;
        return centroid_vec; 
    }

    //TODO: Change to faster method this is high computational burden 
    // 1) K means, 2) Greedy 3) DBScan
    // K means is told that much faster than DBScan 
    std::vector<Eigen::Vector2d>
    Lidar_sensor::frontier_clustering(std::vector<Eigen::Vector2d> frontier_pts){
        std::vector<std::vector<Eigen::Vector2d> > clustered_frontiers; //Clustered_frontiers with index 
        // Initial guess
        // Eigen::Vector2d center = frontier_pts.front();
        //Sort frontier points w.r.t. position (x first, then y.)
        //Cut frontier points into clustering points. Based on threshold radius. 
        sort(frontier_pts.begin(), frontier_pts.end(), grid_map::compare);
        // grid_map::Print_vec(frontier_pts);
        // cout << frontier_pts.size() << endl; 
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
                // cout << "Center x: " << cur_center(0,0) << " y: " << cur_center(1,0) << " distance: " << dist(cur_center, frontier_pts.at(i)) << endl; 
                Num = 1;
                cur_center = frontier_pts.at(i);
                cur_cluster.push_back(cur_center);
            }
        }

        std::random_device rd;
        std::mt19937 gen(rd());


        std::vector<Eigen::Vector2d> selected_ft; 
        for(int i=0; i<clustered_frontiers.size(); i++){
            std::uniform_int_distribution<int> dis(0, clustered_frontiers.at(i).size()-1);
            int select = dis(gen); 
            selected_ft.push_back(clustered_frontiers.at(i).at(select));
        }
        clustered_fts_ = selected_ft;
        return selected_ft;
    }


    /**
     * @brief Get shortest path's distance of clustered frontier points. 
     * 
     * @param pos_euc 
     * @return vector<double> 
     */
    vector<double> Lidar_sensor::get_ft_distance(Eigen::Vector2d& pos_euc){
        vector<double> distance_vec; 

        grid_map::Position pos_grid = grid_map::euc_to_gridref(pos_euc, map_size_);
        // grid_map::Position pos_grid = pos_euc; 
        vector<vector<Eigen::Vector2d> > path; 
        grid_map::Index cur_index; 
        belief_map_.getIndex(pos_grid, cur_index);

        grid_map::Index frontier_index; 
        std::cout << "Before JPS" << std::endl; 
        try
        {
            for(int i=0; i<clustered_fts_.size(); i++){
                std:: cout << clustered_fts_.at(i).transpose() << std::endl; 
                grid_map::Position fts_pos_grid; fts_pos_grid = grid_map::euc_to_gridref(clustered_fts_.at(i), map_size_);
                belief_map_.getIndex(fts_pos_grid, frontier_index);
                Planner::SFC sfc(belief_map_, frontier_index, cur_index);

                std::vector<grid_map::Position> path_grid;
                path_grid = sfc.JPS_Path();
                path.push_back(path_grid);
            }
        }
        catch(const std::exception& e)
        {
            std::cout << "RunTime JPS Error" <<std::endl;
            std::cerr << e.what() << '\n';
            
        }
        std::cout << "Completed JPS" << std::endl;

        // vector<vector<Eigen::Vector2d> > path = get_JPS_Path(pos_euc);
        for(int i=0; i<path.size(); i++)
        {
            double distance = 0.0; 
            for(int j=0; j<path.at(i).size()-1; j++){
                distance = distance + ((path[i][j+1][0]- path[i][j][0])*(path[i][j+1][0]- path[i][j][0])
                                     + (path[i][j+1][1]- path[i][j][1])*(path[i][j+1][1]- path[i][j][1]));
            }
            distance_vec.push_back(distance);
        }
        std::cout << "During Assigning" << std::endl; 
        return distance_vec;
    }

    /**
     * @brief Generate vector of SFCs with respect to given selected frontier vectors. 
     * 
     * @param pos : euc reference
     */
    void Lidar_sensor::construct_SFC(Eigen::Vector2d& pos_euc)
    {   
        grid_map::Position pos_grid = grid_map::euc_to_gridref(pos_euc, map_size_);
        // std::vector<Eigen::Vector2d> obs_grid = generate_obs_grid(pos_euc);
        /**
         * Generate Vector of SFCs  
        **/
        grid_map::Index cur_index; 
        belief_map_.getIndex(pos_grid, cur_index);
        grid_map::Index frontier_index; 
        for(int i=0; i<selected_fts_.size(); i++){
            grid_map::Position ft_pos_gride; ft_pos_gride = grid_map::euc_to_gridref(selected_fts_.at(i), map_size_);
            belief_map_.getIndex(ft_pos_gride, frontier_index);
            Planner::SFC sfc(belief_map_, frontier_index, cur_index);
            // cout << "SFC Before" << endl; 
            sfc.generate_SFC();
            // cout << "Get Corridor Before" << endl; 
            vec_E<Polyhedron<2>> cur_sfc = sfc.get_corridor();
            sfc_ft_pair_.push_back(make_pair(cur_sfc, selected_fts_.at(i)));
        }
        
        //Should return the pair between frontier cell & SFC block.
    }

    void Lidar_sensor::construct_SFC_jwp(Eigen::Vector2d& pos_euc){
        grid_map::Position pos_grid = grid_map::euc_to_gridref(pos_euc, map_size_);
        // grid_map::Position pos_grid = pos;
        // std::vector<Eigen::Vector2d> obs_grid = generate_obs_grid(pos_euc);
        /**
         * Generate Vector of SFCs  
        **/
        grid_map::Index cur_index; 
        belief_map_.getIndex(pos_grid, cur_index);
        grid_map::Index frontier_index; 
 
        for(int i=0; i<selected_fts_.size(); i++){
            std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

            grid_map::Position ft_pos_grid; ft_pos_grid = grid_map::euc_to_gridref(selected_fts_.at(i), map_size_);
            belief_map_.getIndex(ft_pos_grid, frontier_index);
            // cout << "Occupancy Value At Start: " << belief_map_.at("base", cur_index) << endl; 
            // cout << "Occupancy Value At Goal: " << belief_map_.at("base", frontier_index) << endl; 

            std::chrono::duration<double> first = std::chrono::system_clock::now() - start;
            std::cout << "SFC_First : " << first.count() << " seconds" << std::endl;

            Planner::SFC sfc(belief_map_, frontier_index, cur_index);

            std::chrono::duration<double> second = std::chrono::system_clock::now() - start;
            std::cout << "SFC_Second : " << second.count() - first.count() << " seconds" << std::endl;
            // cout << "Start Generating..." << endl; 
            sfc.generate_SFC_jwp();
            std::chrono::duration<double> third = std::chrono::system_clock::now() - start;
            std::cout << "SFC_Third : " << third.count() - second.count() << " seconds" << std::endl;

            auto cur_sfc = sfc.get_corridor_jwp();
            std::chrono::duration<double> fourth = std::chrono::system_clock::now() - start;
            std::cout << "SFC_Fourth : " << fourth.count() - third.count() << " seconds" << std::endl;

            // cout << "Generated SFC" << endl; 
            cout << "Size of SFC: " << cur_sfc.size() << endl; 
            
            // for(int j=0; j<cur_sfc.size(); j++){    
            //     cout << "x : [ " << cur_sfc[j][0] << ", " << cur_sfc[j][2] << "]" << "y: [ " << cur_sfc[j][1] << ", " << cur_sfc[j][3] << "]" << endl; 
            // }
            sfc_jwp_.push_back(cur_sfc);
        }
    }

    std::vector<Eigen::Vector2d> Lidar_sensor::generate_obs_grid(Eigen::Vector2d pos)
    {
        grid_map::Position pos_grid = grid_map::euc_to_gridref(pos, map_size_);
        /**
         * Generate obtacle vector. 
        **/
    //    cout << "Obstacle Vector" << endl; 
        std::vector<Eigen::Vector2d> obs_grid; //Obstacle points based on grid reference. 

        //Set submap iterator
        Eigen::Vector2d top_left(pos(0,0) - (submap_length_(0,0)/2.0), pos(1,0) + submap_length_(1,0)/2.0); //TOp left corner of submap
        Eigen::Vector2d bot_right(pos(0,0) + submap_length_(0,0)/2.0, pos(1,0) - submap_length_(1,0)/2.0);
        grid_map::Position p3 = grid_map::euc_to_gridref(top_left, map_size_);
        grid_map::Position p4 = grid_map::euc_to_gridref(bot_right, map_size_);

        grid_map::Index i3; belief_map_.getIndex(p3, i3); //Find top left index
        grid_map::Index i4; belief_map_.getIndex(p4, i4); //Find top left index
        grid_map::Size submap_size = i4 - i3;

        
        for(grid_map::SubmapIterator it(belief_map_, i3, submap_size); !it.isPastEnd(); ++it)
        {   
            auto index = *it; 
            int int_idx = map_size_x_*index(0,0) + index(1,0);
            auto itr_set = obstacles_.find(int_idx); //Find current index in obstacle unordered set. 
            if(itr_set !=obstacles_.end()){ //Current index is in obstacle set 
                grid_map::Position obst_pos; 
                belief_map_.getPosition(*it, obst_pos);
                obs_grid.push_back(obst_pos);
            }
        }
        // cout << "End Iterator" << endl; 
        return obs_grid; 
    }

    vector<vector<Eigen::Vector2d> > Lidar_sensor::get_JPS_Path(Eigen::Vector2d& pos_euc)
    {  
        grid_map::Position pos_grid = grid_map::euc_to_gridref(pos_euc, map_size_);
        // grid_map::Position pos_grid = pos_euc; 
        vector<vector<Eigen::Vector2d> > total_path_grid; 
        grid_map::Index cur_index; 
        belief_map_.getIndex(pos_grid, cur_index);

        grid_map::Index frontier_index; 

        for(int i=0; i<selected_fts_.size(); i++){
            grid_map::Position fts_pos_grid; fts_pos_grid = grid_map::euc_to_gridref(selected_fts_.at(i), map_size_);
            belief_map_.getIndex(fts_pos_grid, frontier_index);
            Planner::SFC sfc(belief_map_, frontier_index, cur_index);

            std::vector<grid_map::Position> path_grid;
            path_grid = sfc.JPS_Path();
            total_path_grid.push_back(path_grid);
        }
        return total_path_grid;         
        //Should return the pair between frontier cell & SFC block.
    }

}