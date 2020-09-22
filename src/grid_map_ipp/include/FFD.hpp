#ifndef FFD_HPP
#define FFD_HPP

#include "ros/ros.h"
#include <cstdlib> // Needed for rand()
#include <ctime>
#include <queue>
#include <vector>
#include <map>
#include <algorithm>
// #include "nav_msgs/OccupancyGrid.h"
#include "grid_map_core/GridMap.hpp"


//using namespace std;

namespace grid_map{
    struct MyPoint{
    int x;
    int y;
    };

    struct Line{
    std::vector<MyPoint> points;
    };


    class Ft_Detector{
        public:
        Ft_Detector(){}
        // vector<Frontier> frontiersDB, 
        std::vector<std::vector<int> > FFD( MyPoint pose, std::vector<MyPoint> lr, const nav_msgs::OccupancyGrid& map, int map_height, int map_width);
        std::vector<MyPoint> Sort_Polar( std::vector<MyPoint> lr, MyPoint pose);
        Line Get_Line( MyPoint prev, MyPoint curr );

    };
}


#endif