#include <SFC/SFC.hpp>
#include <SFC/JPS.h>

//Obstacle vector is given with respect to the grid reference frame. 
void Planner::SFC::generate_SFC(std::vector<Eigen::Vector2d>& obs_grid)
{
    Planner::JumpPointSearch jps;
    //Map fit to JPS form. 
    //Get solution from JPS;  --> JPS on Index of each grid. 
    //sol = jps.jump_point_search();
    //

    // Set obstacles
    //Transform obs -> obstacles 
    vec_Vec2f obstacles;


    // How to get obstacle data of submap?  
    // Store obstacle points in hash map? 
    // Then search points within boundary of agent's current pose. 

    // if(!read_obs<2>(argv[1], obs)) {
    //     printf(ANSI_COLOR_RED "Cannot find input file [%s]!\n" ANSI_COLOR_RESET,
    //         argv[1]);
    //     return -1;
    // }
    // Set map size
    const Vec2f origin(-2, -2);
    const Vec2f range(4, 4);

    // Path to dilate
    vec_Vec2f path;
    path.push_back(Vec2f(-1.5, 0.0));
    path.push_back(Vec2f(1.5, 0.3));

    // Initialize SeedDecomp2D
    IterativeDecomp2D decomp(origin, range);
    decomp.set_obs(obstacles);
    decomp.set_local_bbox(Vec2f(2, 2));
    decomp.dilate_iter(path, 5, 0.3, 0.0);
}