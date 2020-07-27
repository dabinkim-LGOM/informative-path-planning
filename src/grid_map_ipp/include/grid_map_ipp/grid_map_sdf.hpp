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
        grid_map::GridMap map_;
        grid_map::SignedDistanceField sdf_field_;
        

        public:
        GridMapSDF(double buffer, grid_map::GridMap belief_map)
        : buffer_(buffer)
        {
            // grid_map::ObstacleGridConverter converter(map_size_x, map_size_y, num_obstacle, obstacles);
            // map_ = converter.GridMapConverter();
            map_ = belief_map;
        }

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