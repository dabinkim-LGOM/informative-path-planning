#include "grid_map_core/GridMap.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ipp/wavefront_frontier_detection.hpp>
#include <grid_map_ipp/ObstacleGridConverter.hpp>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <iostream>
#include <Eigen/Dense>
#include <list>
#include <vector>
#include <ros/ros.h>
#include <typeinfo>

using namespace std; 
using namespace grid_map;

visualization_msgs::Marker generate_marker(geometry_msgs::Point pt, double goal_x, double goal_y, int iter)
  {
      visualization_msgs::Marker Marker_lane;
      Marker_lane.id = iter;
      Marker_lane.type = visualization_msgs::Marker::SPHERE;
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
  
      pt.x = pt.x - goal_x;
      pt.y = pt.y - goal_y;
      // Marker_lane.header.stamp = ros::Time();
      Marker_lane.pose.position.x = pt.x;
      Marker_lane.pose.position.y = pt.y;
      Marker_lane.pose.position.z = pt.z;
      return Marker_lane;
 }


int main(int argc, char** argv)
{
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
    

    grid_map::ObstacleGridConverter converter(100.0, 100.0, 5, obstacles);
    grid_map::GridMap gt_map = converter.GridMapConverter();

    
    for(int i=0; i< 50; i++){
        for(int j=0; j<50; j++){
            grid_map::Index idx(i,j);
            gt_map.at("base", idx) = 0.5;
        }
    }

    nav_msgs::OccupancyGrid occ_grid = converter.OccupancyGridConverter(gt_map);
    grid_map::Frontier ft;
    grid_map::Index idx(70,70);
    vector<vector<grid_map::Index> > frontier_vec = ft.wfd(gt_map, idx);

    vector<visualization_msgs::Marker> marker_vec;
    cout << "Size of vector" << frontier_vec.size() << " " << endl;
    int num = 0;
    for(int i=0; i< frontier_vec.size(); i++){
        for(int j=0; j<frontier_vec.at(i).size(); j++){
            num++;
            grid_map::Index idx = frontier_vec.at(i).at(j);
            grid_map::Position pos;
            gt_map.getPosition(idx, pos);
            geometry_msgs::Point pt; 
            pt.x = pos(0,0); pt.y = pos(1,0); pt.z = 0.0;
            ROS_INFO("%d th Frontier point %f, %f", i*frontier_vec.size()+j, pt.x, pt.y);
            visualization_msgs::Marker marker = generate_marker(pt, 0.0, 0.0, num);
            marker_vec.push_back(marker);
        }
    }

    ros::init(argc, argv, "test_frontier");
    ros::NodeHandle nh("");
    ros::Publisher pub = nh.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
    ros::Rate rate(30.0);
    ros::Publisher pub_occ = nh.advertise<nav_msgs::OccupancyGrid>("occu_grid", 1, true);
    ros::Publisher pub_vis_ft = nh.advertise<visualization_msgs::Marker>("frontier", 100, true);
    grid_map::Size size = gt_map.getSize();
    cout<< size(0,0) << endl;



    while(nh.ok())
    {
        ros::Time time = ros::Time::now();
        // grid_map_msgs::GridMap message;
        nav_msgs::OccupancyGrid occ_message;
        nav_msgs::OccupancyGrid &occ_m = occ_message;
        
        // GridMapRosConverter::toMessage(gt_map, message);
        GridMapRosConverter::toOccupancyGrid(gt_map, "base", 0.0, 1.0, occ_m);
        // pub.publish(message);
        pub_occ.publish(occ_grid);
        for(int i=0;i<marker_vec.size(); i++){
            pub_vis_ft.publish(marker_vec.at(i));
        }
        rate.sleep();    
    }

    return 0;
}