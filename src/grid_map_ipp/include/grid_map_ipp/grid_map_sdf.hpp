#ifndef GRIDMAPSDF
#define GRIDMAPSDF

#include "grid_map_core/GridMap.hpp"
#include "grid_map_core/iterators/iterators.hpp"
#include "grid_map_sdf/SignedDistanceField.hpp"
#include "grid_map_ipp/grid_map_ipp.hpp"
#include <vector>
#include <Eigen/Dense>
#include <string>

namespace grid_map
{
    class GridMapSDF
    {
        private:
        double buffer_;
        Eigen::Vector2d position_;
        Eigen::Array2d length_; //Length of submap
        grid_map::GridMap map_;
        grid_map::SignedDistanceField sdf_field_;
        

        public:
        GridMapSDF(double buffer, RayTracer::Lidar_sensor lidar, Eigen::Vector2d pos, Eigen::Array2d length)
        : buffer_(buffer)
        {
            // grid_map::ObstacleGridConverter converter(map_size_x, map_size_y, num_obstacle, obstacles);
            // map_ = converter.GridMapConverter();
            position_ = pos;        length_ = length;   bool isSuccess = true;
            grid_map::GridMap full_map = lidar.get_belief_map();
            map_ = full_map.getSubmap(position_, length_, isSuccess);
            cout << isSuccess << endl;
        }

        void set_length(Eigen::Array2d length){length_= length;};
        void set_position(Eigen::Vector2d pos){position_ = pos;};
        void generate_SDF()
        {
            sdf_field_.calculateSignedDistanceField(map_, "base", 1.5);
        }
        grid_map::Vector3 get_GradientValue(RayTracer::Pose& pos)
        {
            sdf_field_.calculateSignedDistanceField(map_, "base", 1.5);
            grid_map::Vector3 gradient;
            gradient = sdf_field_.getDistanceGradientAt(grid_map::Vector3(pos.x, pos.y, 0.0));
            return gradient;
        }
        double get_Distance(RayTracer::Pose& pos)
        {
            sdf_field_.calculateSignedDistanceField(map_, "base", 1.5);
            auto distance = sdf_field_.getDistanceAt(grid_map::Vector3(pos.x, pos.y, 0.0));
            return distance;
        }

    };


}

#endif