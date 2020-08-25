#include <grid_map_ipp/util.hpp>

namespace grid_map
{
        //Transform euclidean (x,y) position value to grid map reference frame
        Eigen::Vector2d euc_to_gridref(Eigen::Vector2d& pos, Eigen::Array2i& map_size)
        {
            //Rotation (-pi/2) w.r.t. z direction
            Eigen::Vector2d grid_pos;
            grid_pos(0) = pos(1) - map_size(0) /2.0;
            grid_pos(1) = -1.0*pos(0) + map_size(1) /2.0;
            // cout << "WHAT3?" << grid_pos(0) << " " << grid_pos(1) << endl;
            return grid_pos;
        }

        Eigen::Vector2d grid_to_eucref(Eigen::Vector2d& pos, Eigen::Array2i& map_size)
        {
            //Rotation (-pi/2) w.r.t. z direction
            Eigen::Vector2d euc_pos;
            euc_pos(0) = map_size(1) /2.0 - pos(1);
            euc_pos(1) = pos(0) + map_size(0) /2.0;
            // cout << "WHAT3?" << grid_pos(0) << " " << grid_pos(1) << endl;
            return euc_pos;
        }

        std::vector<double> gridbox_to_eucbox(std::vector<double>& box, Eigen::Array2i& map_size)
        {
            Eigen::Vector2d euc_first(box[0], box[1]);
            Eigen::Vector2d euc_second(box[2], box[3]);
            Eigen::Vector2d grid_first = grid_to_eucref(euc_first, map_size);
            Eigen::Vector2d grid_second = grid_to_eucref(euc_second, map_size);
            if(grid_second[0]<grid_first[0]){
                double tmp = grid_first[0];
                grid_first[0] = grid_second[0];
                grid_second[0] = tmp; 
            }
            if(grid_second[1]<grid_first[1]){
                double tmp = grid_first[1];
                grid_first[1] = grid_second[1];
                grid_second[1] = tmp; 
            }
            std::vector<double> grid_box;
            grid_box.emplace_back(grid_first[0]);
            grid_box.emplace_back(grid_first[1]);
            grid_box.emplace_back(grid_second[0]);
            grid_box.emplace_back(grid_second[1]);
            return grid_box; 
        }

        void Print_vec(std::vector<Eigen::Vector2d>& frontiers)
        {
            for(int i=0; i<frontiers.size(); i++){
                Eigen::Vector2d cur = frontiers.at(i);
                std::cout << "x: " << cur(0,0) << " y: " << cur(1,0) << std::endl; 
            }
        }

        bool compare(Eigen::Vector2d& t1, Eigen::Vector2d& t2){
            if(t1(0,0)>t2(0,0))
                return false;
            else if(t1(0,0) < t2(0,0))
                return true;
            else{
                if(t1(1,0) > t2(1,0))
                    return false;    
                else
                {
                    return true; 
                }
            }
        }

}