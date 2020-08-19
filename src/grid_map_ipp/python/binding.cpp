#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <grid_map_ipp/grid_map_ipp.hpp>
#include <grid_map_ipp/ObstacleGridConverter.hpp>
#include <grid_map_ipp/grid_map_sdf.hpp>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace grid_map;
using namespace RayTracer;

int add(int i, int j)
{
    return i+j;
}

PYBIND11_MODULE(grid_map_ipp_module, m)
{
    m.doc()= "lidar binding";

    py::class_<Pose>(m, "Pose")
        .def(py::init<double &, double &, double&>());

    py::class_<ObstacleGridConverter>(m, "ObstacleGridConverter")
        .def(py::init<double &, double &, int &,
                      std::vector<Eigen::Array4d> &>())
        .def("grid_map_converter", (nav_msgs::OccupancyGrid (ObstacleGridConverter::*)()) &ObstacleGridConverter::GridMapConverter)
        .def("OccupancyGridConverter",(nav_msgs::OccupancyGrid (ObstacleGridConverter::*)(grid_map::GridMap&)) &ObstacleGridConverter::OccupancyGridConverter);
    
    py::class_<Raytracer>(m, "Raytracer")
        .def(py::init<double &, double &, int &,
                      std::vector<Eigen::Array4d> &>())
        .def("get_grid_map", &Raytracer::get_grid_map);

    py::class_<Lidar_sensor>(m, "Lidar_sensor")
        .def(py::init<double &, double &, 
                      double &, double &, 
                      double &, double &, 
                      double &, double &, Raytracer &>())
        // .def("init_belief_map", &Lidar_sensor::init_belief_map)
        .def("get_measurement", &Lidar_sensor::get_measurement)
        .def("get_belief_map", &Lidar_sensor::get_belief_map)
        .def("frontier_detection", &Lidar_sensor::frontier_detection)
        .def("frontier_clustering", &Lidar_sensor::frontier_clustering)
        .def("selected_fts", &Lidar_sensor::set_selected_frontier)
        .def("get_occ_value", &Lidar_sensor::get_occ_value);
        // .def()
    py::class_<GridMapSDF>(m, "GridMapSDF")
        .def(py::init<double &, Lidar_sensor &, Eigen::Vector2d &, Eigen::Array2d &>())
        .def("set_length", &GridMapSDF::set_length)
        .def("set_position", &GridMapSDF::set_position)
        .def("generate_SDF", &GridMapSDF::generate_SDF)
        .def("is_occupied", &GridMapSDF::is_occupied)
        .def("get_gradient_value", &GridMapSDF::get_GradientValue)
        .def("get_distance", &GridMapSDF::get_Distance);
}