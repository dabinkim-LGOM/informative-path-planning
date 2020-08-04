#ifndef GRIDMAPSDF
#define GRIDMAPSDF

#include "grid_map_core/GridMap.hpp"
#include "grid_map_core/iterators/iterators.hpp"
#include "grid_map_sdf/SignedDistanceField.hpp"
#include "grid_map_ipp/grid_map_ipp.hpp"
#include <vector>
#include <Eigen/Dense>
#include <string>
using namespace std;

namespace grid_map
{
    class GridMapSDF
    {
        private:
        double buffer_;
        Eigen::Vector2d position_;
        Eigen::Array2d length_; //Length of submap
        grid_map::GridMap map_;
        // grid_map::SubmapGeometry submap_;
        grid_map::SignedDistanceField sdf_field_;
        

        public:
        GridMapSDF(double buffer, RayTracer::Lidar_sensor lidar, Eigen::Vector2d pos, Eigen::Array2d length)
        : buffer_(buffer)
        {
            // grid_map::ObstacleGridConverter converter(map_size_x, map_size_y, num_obstacle, obstacles);
            // map_ = converter.GridMapConverter();
            grid_map::GridMap full_map = lidar.get_belief_map();
            grid_map::Size mapsize = full_map.getSize();
            Eigen::Vector2d vec_mapsize(mapsize(0) * full_map.getResolution(), mapsize(1) * full_map.getResolution());
            position_ = pos - vec_mapsize * 0.5;        length_ = length;   bool isSuccess = true;
            
            // grid_map::SubmapGeometry submap(full_map, position_, length_, isSuccess);
            // submap_ = submap; 
            // cout << "Position in C++, x: " << position_(0) <<" y: " << position_(1) << endl;
            map_ = full_map.getSubmap(position_, length_, isSuccess);
            // Or just call
            // map_ = lidar.get_submap(pos, length);
            // if(isSuccess)
            //     cout << "Submap gen success " << endl;
            // else
            //     cout << "Submap gen fail " << endl;

        }

        void set_length(Eigen::Array2d length){length_= length;};
        void set_position(Eigen::Vector2d pos){position_ = pos;};
        void generate_SDF(){   
            try{
                // cout << "Hello" << endl;
                grid_map::Matrix map = map_.get("base");
                float maxHeight = map.maxCoeffOfFinites();
                // cout << "maxHeight " << maxHeight << endl;
                // cout << "size " << map.size() << endl;
                // cout << "rows " << map.rows() << endl;
                // cout << "cols " << map.cols() << endl;
                sdf_field_.calculateSignedDistanceField(map_, "base", 0.0);
            }
            catch(const std::exception& e){
                std::cerr << e.what() << '\n';
            }
        }

        grid_map::Vector3 get_GradientValue(Eigen::Vector2d& pos)
        {
            sdf_field_.calculateSignedDistanceField(map_, "base", 1.5);
            grid_map::Vector3 gradient;
            gradient = sdf_field_.getDistanceGradientAt(grid_map::Vector3(pos(0), pos(1), 0.0));
            return gradient;
        }

        double get_Distance(Eigen::Vector2d& pos)
        {
            sdf_field_.calculateSignedDistanceField(map_, "base", 1.5);
            auto distance = sdf_field_.getDistanceAt(grid_map::Vector3(pos(0), pos(1), 0.0));
            return distance;
        }

        bool is_occupied()
        {   
            try
            {
                // grid_map::SubmapIterator sub_iterator(submap_);
                for(GridMapIterator it(map_); !it.isPastEnd(); ++it){
                    if(map_.at("base", *it) > 0.9 ){
                        return true;
                    }
                }
                return false;
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
            }            
        }

    };


}

#endif