#include <Frontier.hpp>

namespace grid_map{
    void Frontier::wfd_frontier(grid_map::Index& pose)
    {
        queue<grid_ft&> detect_ft;
        queue<grid_ft&> extract_ft;
        grid_map::grid_ft pose_ft(pose);
        grid_map::grid_ft& pose_ref = pose_ft;
        pose_ref.is_MapOpen = true;  //Mark pose as Map-Open-List;
        detect_ft.push(pose_ref);
        
        while(!detect_ft.empty())
        {
            grid_map::grid_ft p = detect_ft.front();
            detect_ft.pop();
            if(p.is_MapClose){
                continue;
            }
            

        }

    }
}