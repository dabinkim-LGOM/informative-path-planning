#include <SFC/SFC.hpp>

using namespace std;
 
vec_E<Polyhedron<2>> Planner::SFC::generate_SFC()
{
    Planner::JumpPointSearch jps;
    //Map fit to JPS form. 
    //Get solution from JPS;  --> JPS on Index of each grid. 
    //sol = jps.jump_point_search();
    //
    // cout << "Before JPS" << endl; 
    std::vector<Planner::Node> jps_result; 
    // cout << "Cur Index: " << cur_index_(0,0) << ", " << cur_index_(1,0) << endl; 
    // cout << "Frontier Goal Index: " << goal_frontier_(0,0) << ", " << goal_frontier_(1,0) << endl; 
    
    Planner::Node start_node(cur_index_, 0.0, 0.0, 0, 0); 
    Planner::Node goal_node(goal_frontier_, 0.0, 0.0, 0, 0);
    jps_result = jps.jump_point_search(belief_map_, start_node, goal_node);
    // cout << "After JPS" << endl; 

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
    // vec_Vec2f obstacles;
    // Vec2f cur_obstacle;
    // for(int i=0; i<obs_grid.size(); i++){
    //     cur_obstacle(0,0) = (float) (obs_grid.at(i))(0,0);
    //     cur_obstacle(1,0) = (float) (obs_grid.at(i))(1,0);
    //     obstacles.push_back(cur_obstacle);
    // }


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

    // Initialize SeedDecomp2D
    EllipsoidDecomp2D decomp(origin, range);
    // decomp.set_obs(obstacles);
    decomp.set_local_bbox(Vec2f(2, 2));
    decomp.dilate(recon_jps_path, 0.0);
    
    auto SFC = decomp.get_polyhedrons();
    // std::cout << SFC.at(0).hyperplanes
    // Corridor_ = SFC;     
    return SFC; 
}


void Planner::SFC::generate_SFC_jwp()
{
    // Planner::JumpPointSearch jps;
    // //Map fit to JPS form. 
    // //Get solution from JPS;  --> JPS on Index of each grid. 
    // //sol = jps.jump_point_search();
    
    // // cout << "Before JPS" << endl; 
    // std::vector<Planner::Node> jps_result; 
    // // cout << "Cur Index: " << cur_index_(0,0) << ", " << cur_index_(1,0) << endl; 
    // // cout << "Frontier Goal Index: " << goal_frontier_(0,0) << ", " << goal_frontier_(1,0) << endl; 
    
    // Planner::Node start_node(cur_index_, 0.0, 0.0, 0, 0); 
    // Planner::Node goal_node(goal_frontier_, 0.0, 0.0, 0, 0);
    // jps_result = jps.jump_point_search(belief_map_, start_node, goal_node);
    // cout << "After JPS" << endl; 

    //Reconstruct path from grid index to grid-ref double values. 
    std::vector<Eigen::Vector2d> recon_jps_path; 
    // grid_map::Position pos;
    // // Eigen::Vector2d float_pos; 
    // for(int i=0; i<jps_result.size(); i++){
    //     belief_map_.getPosition(jps_result.at(i).idx_, pos);
    //     // float_pos(0,0) = pos(0,0);
    //     // float_pos(1,0) = pos(1,0);
        
    //     recon_jps_path.push_back(pos);
    // }
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    recon_jps_path = JPS_Path();
    std::chrono::duration<double> third = std::chrono::system_clock::now() - start;
    std::cout << "SFC_3_1 : " << third.count() << " seconds" << std::endl;
    // cout << "RECON_JPS_PATH" << endl; 
    // for(int i=0; i<recon_jps_path.size(); i++){
    //         cout << recon_jps_path.at(i).transpose() << endl; 
    // }
    
    bool gen_box = updateObsBox(recon_jps_path);
    std::chrono::duration<double> fourth = std::chrono::system_clock::now() - start;
    std::cout << "SFC_3_2 : " << fourth.count() - third.count() << " seconds" << std::endl;
}

//Grid reference frame
std::vector<Eigen::Vector2d> Planner::SFC::JPS_Path()
{
    Planner::JumpPointSearch jps;

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



std::vector<double> Planner::SFC::expand_box(std::vector<double> &box, double margin) {
    std::vector<double> box_cand, box_update;
    std::vector<int> axis_cand{0, 1, 2, 3};

    int i = -1;
    int axis;
    while (!axis_cand.empty()) {
        
        box_cand = box;
        box_update = box;
        // cout << "TRUE OF FALSE: " << (!isObstacleInBox(box_update, margin) && isBoxInBoundary(box_update)) << endl; 
        // cout << "OBSTACLE: " << !isObstacleInBox(box_update, margin)  << endl; 
        // cout << "BOUNDARY: " << isBoxInBoundary(box_update) << endl; 
        //check update_box only! update_box + current_box = cand_box
        // for(int i=0; i<4; i++){
            

            while (!isObstacleInBox(box_update, margin) && isBoxInBoundary(box_update)) {
                i++;
                if (i >= axis_cand.size()) {
                    i = 0;
                }
                axis = axis_cand[i];

                //update current box
                // cout << "IN THE WHILE LOOP!" << endl; 
                box = box_cand;
                box_update = box_cand;

                //expand cand_box and get updated part of box(update_box)
                if (axis < 2) {
                    box_update[axis + 2] = box_cand[axis];
                    box_cand[axis] = box_cand[axis] - box_xy_res;
                    box_update[axis] = box_cand[axis];
                    // cout << "AXIS: " << axis << " Val: " << box_update[axis] << endl; 
                    // if(isObstacleInBox(box_update, margin))
                    //     cout << "OBSTACLE IN BOX" << endl; 
                    // if(!isBoxInBoundary(box_update))
                    //     cout << "BOX IN BOUNDARY" << endl; 
                } else {
                    box_update[axis - 2] = box_cand[axis];
                    box_cand[axis] = box_cand[axis] + box_xy_res;
                    box_update[axis] = box_cand[axis];
                    // cout << "AXIS: " << axis << " Val: " << box_update[axis] << endl; 
                    // if(isObstacleInBox(box_update, margin))
                    //     cout << "OBSTACLE IN BOX" << endl; 
                    // if(!isBoxInBoundary(box_update))
                    //     cout << "BOX IN BOUNDARY" << endl; 
                }

            }

            axis_cand.erase(axis_cand.begin()+i);
            if (i > 0) {
                i--;
            } else {
                i = axis_cand.size() - 1;
            }
        
    }
    box = box_cand; 
    return box;
}

/**
 * Initial trajectory is given as index value. 
 * */
bool Planner::SFC::updateObsBox(std::vector<Eigen::Vector2d> initTraj) {
    double x_next, y_next, dx, dy;

    // //Reduce initTraj's size. (Prune redundant path)
    // Eigen::Vector2d tmp_grad(0,0);
    // Eigen::Vector2d next_grad(0,0);
    // for(int i=0; i<initTraj.size();i++){
    //     tmp_grad = initTraj[i+1] - initTraj[i]; tmp_grad = tmp_grad/(tmp_grad.squaredNorm());
    //     next_grad = initTraj[i+2] - initTraj[i+1]; next_grad = next_grad/(next_grad.squaredNorm());
    //     if(tmp_grad[0] ==next_grad[0] && tmp_grad[1] == next_grad[1]){
    //         initTraj.erase(initTraj.begin()+(i+1));
    //         i--;
    //     }
    //     if(i==initTraj.size()-1)
    //         break; 
    // }

    // cout << "RECON_JPS_PATH" << endl; 
    // for(int i=0; i<initTraj.size(); i++){
    //         cout << initTraj.at(i).transpose() << endl; 
    // }


    std::vector<double> box_prev{0, 0, 0, 0};


    for (int i = 0; i < initTraj.size() - 1; i++) {
        auto state = initTraj[i];
        double x = state(0,0);
        double y = state(1,0);
        // cout << "STATE: " << i << " " <<  state.transpose() << endl; 
        // if(i+1==initTraj.size()-1)
            // cout << "STATE: " << i+1 << " " <<  initTraj[i+1].transpose() << endl; 
        
        std::vector<double> box;
        auto state_next = initTraj[i + 1];
        x_next = state_next(0,0);
        y_next = state_next(1,0);

        // Eigen::Vector2d pos_next(x_next, y_next);
        // if (isPointInBox(pos_next, box_prev)) {
        //     continue;
        // }

        // Initialize box
        box.emplace_back(round(std::min(x, x_next) / box_xy_res) * box_xy_res);
        box.emplace_back(round(std::min(y, y_next) / box_xy_res) * box_xy_res);
        box.emplace_back(round(std::max(x, x_next) / box_xy_res) * box_xy_res);
        box.emplace_back(round(std::max(y, y_next) / box_xy_res) * box_xy_res);

        if (isObstacleInBox(box, margin_)) {
            std::cout << "Corridor: Invalid initial trajectory. Obstacle invades initial trajectory." << std::endl;
            std::cout << "Corridor: x " << x << ", y " << y << std::endl;

            bool debug =isObstacleInBox(box, margin_);
            return false;
        }
        box = expand_box(box, margin_);

        cout <<"[BOX] x : [ " << box[0] << ", " << box[2] << "]" << "y: [ " << box[1] << ", " << box[3] << "]" << endl; 
        Corridor_jwp_.emplace_back(box);

        box_prev = box;
    }



    // timer.stop();
    // ROS_INFO_STREAM("Corridor: SFC runtime=" << timer.elapsedSeconds());
    return true;
}

/**
 * Check wheter BOX has obstacle inside. 
 * BOX is given in index values. 
 * */
bool Planner::SFC::isObstacleInBox(const std::vector<double> &box, double margin) {
    double x, y;

    // for(int i=0; i<obs_grid.size(); i++){
    //     auto obs_grid_pos = obs_grid.at(i); 
    //     grid_map::Index obs_grid; 
    //     belief_map_.getIndex(obs_grid_pos, obs_grid);
    //     x = obs_grid(0,0); y = obs_grid(1,0);
    //     if( box[0]-SP_EPSILON < x && x < box[2]+SP_EPSILON )
    //         return true; 
    //     if( box[1]-SP_EPSILON < y && y < box[3]+SP_EPSILON )
    //         return true;
    // }
    // return false; 

    int count1 = 0;
    for (double i = box[0]; i < box[2] + SP_EPSILON; i += 2*box_xy_res) {
        int count2 = 0;
        for (double j = box[1]; j < box[3] + SP_EPSILON; j += 2*box_xy_res) {
            int count3 = 0;

                x = i + SP_EPSILON;
                if (count1 == 0 && box[0] > world_x_min + SP_EPSILON) {
                    x = box[0] - SP_EPSILON;
                }
                y = j + SP_EPSILON;
                if (count2 == 0 && box[1] > world_y_min + SP_EPSILON) {
                    y = box[1] - SP_EPSILON;
                }

                Eigen::Vector2d cur_point(x, y); grid_map::Index cur_idx; 
                belief_map_.getIndex(cur_point, cur_idx);
                
                if(cur_idx(0,0)>=(belief_map_.getSize())(0,0) || cur_idx(1,0)>=(belief_map_.getSize())(1,0)
                    || cur_idx(0,0)<0 || cur_idx(1,0)<0)
                    return true; 
                if(belief_map_.at("base", cur_idx) > 0.15){
                    return true; 
                }
                // float dist = distmap_obj.get()->getDistance(cur_point);
                // if (dist < margin - SP_EPSILON) {
                //     return true;
                // }
            
            count2++;
        }
        count1++;
    }
    return false; 
}

bool Planner::SFC::isBoxInBoundary(const std::vector<double> &box) {
    return box[0] > world_x_min + SP_EPSILON &&
            box[1] > world_y_min + SP_EPSILON &&
            box[2] < world_x_max - SP_EPSILON &&
            box[3] < world_y_max - SP_EPSILON;
}

bool Planner::SFC::isPointInBox(const grid_map::Position &point,
                    const std::vector<double> &box) {
    return point(0,0) > box[0] - SP_EPSILON &&
            point(1,0) > box[1] - SP_EPSILON &&
            point(0,0) < box[2] + SP_EPSILON &&
            point(1,0) < box[3] + SP_EPSILON;
}



void Planner::SFC::visualize_SFC(vec_E<Polyhedron<2>>& SFC)
{

}