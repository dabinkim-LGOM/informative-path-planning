#include <grid_map_ipp/ObstacleGridConverter.hpp>

using namespace std;

namespace grid_map
{
    grid_map::GridMap ObstacleGridConverter::GridMapConverter()
    {
        vector<string> name;
        name.push_back("base");

        grid_map::GridMap gt_map(name);
        gt_map.setFrameId("map");
        cout << map_size_x_ << endl;
        cout << map_size_y_ << endl;
        
        gt_map.setGeometry(Length(map_size_x_, map_size_y_), 1.00);
        gt_map.add("base", 0.0); //Set all values to zero.

        double buffer = 1.0;
        for (GridMapIterator it(gt_map); !it.isPastEnd(); ++it){
            Position position;
            gt_map.getPosition(*it, position);
            // cout << position.x() << endl;
            double x = position.x() + map_size_x_/2.0; 
            double y = position.y() + map_size_y_/2.0;
            bool is_obs = false;

            //Check current pos. is inside any obstacles. If yes, it set the grid map value to 1.0
            for (vector<Eigen::Array4d>::iterator iter=obstacles_.begin(); iter!=obstacles_.end(); iter++){
                Eigen::Array4d size = (*iter);
                if( x > size(0,0) - buffer && y >size(1,0) - buffer ){
                    if( x<size(2,0) + buffer && y < size(3,0) + buffer){
                        is_obs = true;
                    }
                }
            }
            if(is_obs){
                gt_map.at("base", *it) = 1.0; //Obstacle
            }
        }
        return gt_map;
    } 
    
    nav_msgs::OccupancyGrid ObstacleGridConverter::OccupancyGridConverter()
    {
        grid_map::GridMap grid_map = ObstacleGridConverter::GridMapConverter();
        nav_msgs::OccupancyGrid occ_grid;
        GridMapRosConverter::toOccupancyGrid(grid_map, "base", 0.0, 1.0, occ_grid);
        return occ_grid;
    }

}