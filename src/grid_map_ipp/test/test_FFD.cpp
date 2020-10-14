#include "grid_map_core/GridMap.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include <grid_map_msgs/GridMap.h>
// #include <grid_map_ipp/wavefront_frontier_detection.hpp>
#include <grid_map_ipp/FFD.hpp>
#include <grid_map_ipp/ObstacleGridConverter.hpp>
#include <grid_map_ipp/grid_map_ipp.hpp>
// #include <grid_map_ipp/util.hpp>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <iostream>
#include <Eigen/Dense>
#include <list>
#include <vector>
#include <ros/ros.h>
#include <typeinfo>
#include <random>
#include <chrono>


using namespace std; 
using namespace grid_map;

visualization_msgs::Marker generate_marker(vector<geometry_msgs::Point> pt_vec, double goal_x, double goal_y, int iter)
  {
    visualization_msgs::Marker Marker_lane;
    Marker_lane.id = iter;
    Marker_lane.type = visualization_msgs::Marker::SPHERE_LIST;
    Marker_lane.action = visualization_msgs::Marker::ADD;
    Marker_lane.pose.orientation.w = 1.0;
    Marker_lane.pose.orientation.x = 0.0;
    Marker_lane.pose.orientation.y = 0.0;
    Marker_lane.pose.orientation.z = 0.0;

    Marker_lane.header.frame_id = "map";
    Marker_lane.color.a = 1.0;
    Marker_lane.color.r = 1.0;
    Marker_lane.color.g = 0.0;
    Marker_lane.color.b = 0.0;
    Marker_lane.scale.x = 0.5;
    Marker_lane.scale.y = 0.5;
    Marker_lane.scale.z = 0.5;

    for(int i=0; i<pt_vec.size(); i++){
        geometry_msgs::Point point;
        point.x = pt_vec[i].x - goal_x;
        point.y = pt_vec[i].y - goal_y;
        Marker_lane.points.push_back(point);
    }

    return Marker_lane;
 }

visualization_msgs::Marker generate_marker(vector<geometry_msgs::Point> pt_vec, double r, double g, double b, int iter)
  {
    visualization_msgs::Marker Marker_lane;
    Marker_lane.id = iter;
    Marker_lane.type = visualization_msgs::Marker::SPHERE_LIST;
    Marker_lane.action = visualization_msgs::Marker::ADD;
    Marker_lane.pose.orientation.w = 1.0;
    Marker_lane.pose.orientation.x = 0.0;
    Marker_lane.pose.orientation.y = 0.0;
    Marker_lane.pose.orientation.z = 0.0;

    Marker_lane.header.frame_id = "map";
    Marker_lane.color.a = 1.0;
    Marker_lane.color.r = r;
    Marker_lane.color.g = g;
    Marker_lane.color.b = b;
    Marker_lane.scale.x = 0.5;
    Marker_lane.scale.y = 0.5;
    Marker_lane.scale.z = 0.5;

    for(int i=0; i<pt_vec.size(); i++){
        geometry_msgs::Point point;
        point.x = pt_vec[i].x;
        point.y = pt_vec[i].y;
        Marker_lane.points.push_back(point);
    }

    return Marker_lane;
 }


int main(int argc, char** argv)
{   
    //ROS 
    ros::init(argc, argv, "test_frontier");
    ros::NodeHandle nh("");
    ros::Publisher pub = nh.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
    ros::Rate rate(2.0);
    ros::Publisher pub_belief = nh.advertise<nav_msgs::OccupancyGrid>("belief_map", 100, true);
    ros::Publisher pub_occ = nh.advertise<nav_msgs::OccupancyGrid>("occu_grid", 100, true);
    ros::Publisher pub_vis_ft = nh.advertise<visualization_msgs::Marker>("frontier", 100, true);
    ros::Publisher pub_vis_contour = nh.advertise<visualization_msgs::Marker>("contour", 100, true);
    ros::Publisher pub_vis_sorted = nh.advertise<visualization_msgs::Marker>("sorted", 100, true);

    //Map Generation 
    int num_box = 5;
    double dim_x = 5.0; double dim_y = 5.0;
    std::list<std::pair<double,double> > center;
    pair<double, double> center1 = make_pair(10.0, 20.0);
    pair<double, double> center2 = make_pair(50.0, 30.0);
    pair<double, double> center3 = make_pair(60.0, 80.0);
    pair<double, double> center4 = make_pair(30.0, 60.0);
    pair<double, double> center5 = make_pair(70.0, 90.0);
    
    center.push_back(center1);
    center.push_back(center2);
    center.push_back(center3);
    center.push_back(center4);
    center.push_back(center5);
    list<pair<double, double>>::iterator iter = center.begin();
    vector<Eigen::Array4d> obstacles;
    for(iter=center.begin(); iter!=center.end(); iter++)
    {
        Eigen::Array4d point((*iter).first - dim_x / 2.0, (*iter).second - dim_y / 2.0, (*iter).first + dim_x / 2.0, (*iter).second + dim_y / 2.0);
        obstacles.push_back(point);
    }
    
    //ObstacleGridConverter : Conventional x, y coordinate 
    grid_map::ObstacleGridConverter converter(100.0, 100.0, 5, obstacles);
    grid_map::GridMap gt_map = converter.GridMapConverter();
    nav_msgs::OccupancyGrid occ_grid = converter.OccupancyGridConverter(gt_map);
    nav_msgs::OccupancyGrid belief_grid; 
    

    // for(int i=0; i< 50; i++){
    //     for(int j=0; j<50; j++){
    //         grid_map::Index idx(i,j);
    //         gt_map.at("base", idx) = 0.5;
    //     }
    // }

    //Sensor generation 
    double range_max = 10.0; double range_min = 0.5; 
    double hangle_max = 180; double hangle_min = -180; double angle_resol = 1.0;
    double resol = 1.0;

    RayTracer::Raytracer raytracer(100.0, 100.0, 5, obstacles);
    RayTracer::Lidar_sensor lidar(range_max, range_min, hangle_max, hangle_min, angle_resol, 100.0, 100.0, resol, raytracer);

    int x_idx; int y_idx;
    nh.param("x_idx", x_idx, 50);
    nh.param("y_idx", y_idx, 50);
    grid_map::Index idx(51,50);


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 20);

    grid_map::Position cur_pos; 
    gt_map.getPosition(idx, cur_pos);
    grid_map::Size size = gt_map.getSize();
    cur_pos = grid_map::grid_to_eucref(cur_pos, size);
    // cur_pos(0,0) = cur_pos(0,0) + 50.0; 
    // cur_pos(1,0) = cur_pos(1,0) + 50.0; 

    ROS_INFO("ROS LOOP STARTED");
    while(nh.ok())
    {
        ros::Time time = ros::Time::now();
        grid_map::GridMap belief = lidar.get_belief_map();
        // belief_grid = converter.GridMapConverter(belief, "base", 0.0, 1.0, )
        GridMapRosConverter::toOccupancyGrid(belief, "base", 0.0, 1.0, belief_grid);
        // grid_map_msgs::GridMap message;
        nav_msgs::OccupancyGrid occ_message;
        nav_msgs::OccupancyGrid &occ_m = occ_message;
        
        // GridMapRosConverter::toMessage(gt_map, message);
        GridMapRosConverter::toOccupancyGrid(gt_map, "base", 0.0, 1.0, occ_m);
        // pub.publish(message);

        
        RayTracer::Pose position(cur_pos[0], cur_pos[1], 0.0);
        
        
        // cur_pose.x = cur_pos[0]; cur_pose.y = cur_pos[1]; cur_pose.yaw = 0.0; 
        // lidar.set_belief_map(gt_map);
        lidar.get_measurement(position);

        std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
        std::vector<Eigen::Vector2d > frontier_pos = lidar.FFD(cur_pos);
        std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds currentSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        cout << "[FFD] spent " << currentSeconds.count() << "ms" << endl;

        std::vector<Eigen::Vector2d> contour_pos = lidar.get_contour();
        std::vector<Eigen::Vector2d> sorted_pos = lidar.get_sorted();

        visualization_msgs::Marker marker_vec;
        // cout << "Size of vector" << frontier_vec.size() << " " << endl;
        int num = 0;
        // cout << "Size of cluster" << clustered_frontiers.size() << " " << endl; 
        
        vector<geometry_msgs::Point> pt_vec; 
        std::cout << "Size of frontier pos " << frontier_pos.size() << std::endl; 
        for(int i=0; i< frontier_pos.size(); i++){
            double r = dis(gen)/20.0; 
            // for(int j=0; j<frontier_pos.at(i).size(); j++){
                num++;
                Eigen::Vector2d pos = frontier_pos.at(i);
                // grid_map::Index idx = frontier_vec.at(i).at(j);
                // grid_map::Position pos;
                // gt_map.getPosition(idx, pos);
                geometry_msgs::Point pt; 
                pt.x = pos(0,0); pt.y = pos(1,0); pt.z = 0.0;
                // ROS_INFO("%d th Frontier point %f, %f in Index %d, %d", i*frontier_vec.size()+j, pt.x, pt.y, idx(0,0), idx(1,0));
                pt_vec.push_back(pt);
            // }
        }

        visualization_msgs::Marker marker = generate_marker(pt_vec, 0.0, 0.0, num);


        vector<geometry_msgs::Point> contour_vec; 
        // std::cout << "Size of frontier pos " << frontier_pos.size() << std::endl; 
        for(int i=0; i< contour_pos.size(); i++){
            double r = dis(gen)/20.0; 
            // for(int j=0; j<frontier_pos.at(i).size(); j++){
                num++;
                Eigen::Vector2d pos = contour_pos.at(i);
                // grid_map::Index idx = frontier_vec.at(i).at(j);
                // grid_map::Position pos;
                // gt_map.getPosition(idx, pos);
                geometry_msgs::Point pt; 
                pt.x = pos(0,0); pt.y = pos(1,0); pt.z = 0.0;
                // ROS_INFO("%d th Frontier point %f, %f in Index %d, %d", i*frontier_vec.size()+j, pt.x, pt.y, idx(0,0), idx(1,0));
                contour_vec.push_back(pt);
            // }
        }
        visualization_msgs::Marker contour_marker = generate_marker(contour_vec, 0.0, 1.0, 0.0, 1);

        vector<geometry_msgs::Point> sorted_vec; 
        // std::cout << "Size of frontier pos " << frontier_pos.size() << std::endl; 
        for(int i=0; i< sorted_pos.size(); i++){
            double r = dis(gen)/20.0; 
            // for(int j=0; j<frontier_pos.at(i).size(); j++){
                num++;
                Eigen::Vector2d pos = sorted_pos.at(i);
                // grid_map::Index idx = frontier_vec.at(i).at(j);
                // grid_map::Position pos;
                // gt_map.getPosition(idx, pos);
                geometry_msgs::Point pt; 
                pt.x = pos(0,0); pt.y = pos(1,0); pt.z = 0.0;
                // ROS_INFO("%d th Frontier point %f, %f in Index %d, %d", i*frontier_vec.size()+j, pt.x, pt.y, idx(0,0), idx(1,0));
                sorted_vec.push_back(pt);
            // }
        }
        visualization_msgs::Marker sorted_marker = generate_marker(sorted_vec, 0.0, 0.0, 1.0, 1);

        marker_vec=marker;
        
        pub_occ.publish(occ_grid);
        pub_belief.publish(belief_grid);
        // for(int i=0;i<marker_vec.size(); i++){
        pub_vis_ft.publish(marker_vec);
        pub_vis_contour.publish(contour_marker);
        pub_vis_sorted.publish(sorted_marker);
        // }
        cur_pos[0] = cur_pos[0] + 0.1;
        cur_pos[1] = cur_pos[1] + 0.1;

        rate.sleep();    
    }

    return 0;
}