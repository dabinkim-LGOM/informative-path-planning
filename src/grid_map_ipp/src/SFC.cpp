#include <SFC/SFC.hpp>


//Obstacle vector is given with respect to the grid reference frame. 
vec_E<Polyhedron<2>> Planner::SFC::generate_SFC(std::vector<Eigen::Vector2d>& obs_grid)
{
    Planner::JumpPointSearch jps;
    //Map fit to JPS form. 
    //Get solution from JPS;  --> JPS on Index of each grid. 
    //sol = jps.jump_point_search();
    //
    
    std::vector<Planner::Node> jps_result; 
    Planner::Node start_node(cur_index_, 0.0, 0.0, 0, 0); 
    Planner::Node goal_node(goal_frontier_, 0.0, 0.0, 0, 0);
    jps_result = jps.jump_point_search(belief_map_, start_node, goal_node);
    
    //Reconstruct path from grid index to grid-ref double values. 
    vec_Vec2f recon_jps_path; 
    grid_map::Position pos;
    Eigen::Vector2f float_pos; 
    for(int i=0; i<jps_result.size(); i++){
        belief_map_.getPosition(jps_result.at(i).idx_, pos);
        float_pos(0,0) = (float) pos(0,0);
        float_pos(1,0) = (float) pos(1,0);
        
        recon_jps_path.push_back(pos);
    }

    // Set obstacles
    //Transform obs -> obstacles 
    vec_Vec2f obstacles;
    Vec2f cur_obstacle;
    for(int i=0; i<obs_grid.size(); i++){
        cur_obstacle(0,0) = (float) (obs_grid.at(i))(0,0);
        cur_obstacle(1,0) = (float) (obs_grid.at(i))(1,0);
        obstacles.push_back(cur_obstacle);
    }


    // How to get obstacle data of submap?  
    // Store obstacle points in hash map? 
    // Then search points within boundary of agent's current pose. 

    // if(!read_obs<2>(argv[1], obs)) {
    //     printf(ANSI_COLOR_RED "Cannot find input file [%s]!\n" ANSI_COLOR_RESET,
    //         argv[1]);
    //     return -1;
    // }
    // Set map size
    const Vec2f origin(0.0, 0.0);
    grid_map::Size size_; size_ = belief_map_.getSize();
    float x_size = (float) size_(0,0); float y_size = (float) size_(1,0);
    const Vec2f range(x_size, y_size);

    // Path to dilate
    vec_Vec2f path;
    // path.push_back(Vec2f(-1.5, 0.0));
    // path.push_back(Vec2f(1.5, 0.3));

    // Initialize SeedDecomp2D
    IterativeDecomp2D decomp(origin, range);
    decomp.set_obs(obstacles);
    decomp.set_local_bbox(Vec2f(2, 2));
    decomp.dilate_iter(recon_jps_path, 5, 0.3, 0.0);
    
    vec_E<Polyhedron<2>> SFC = decomp.get_polyhedrons();
    Corridor_ = SFC; 
    return SFC; 
}

//Grid reference frame
std::vector<Eigen::Vector2d> Planner::SFC::JPS_Path()
{
    Planner::JumpPointSearch jps;
    //Map fit to JPS form. 
    //Get solution from JPS;  --> JPS on Index of each grid. 
    //sol = jps.jump_point_search();
    //
    
    std::vector<Planner::Node> jps_result; 
    Planner::Node start_node(cur_index_, 0.0, 0.0, 0, 0); 
    Planner::Node goal_node(goal_frontier_, 0.0, 0.0, 0, 0);
    jps_result = jps.jump_point_search(belief_map_, start_node, goal_node);
    
    std::vector<Eigen::Vector2d> eigen_result;
    for(int i=0; i< jps_result.size(); i++){
        Eigen::Vector2d pos; 
        grid_map::Index idx = jps_result.at(i).idx_;
        belief_map_.getPosition(idx, pos);
        eigen_result.push_back(pos);
    }

    return eigen_result; 
}

void Planner::SFC::visualize_SFC(vec_E<Polyhedron<2>>& SFC)
{

}