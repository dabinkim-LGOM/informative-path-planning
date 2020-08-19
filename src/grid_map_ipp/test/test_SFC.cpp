#include "grid_map_core/GridMap.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ipp/wavefront_frontier_detection.hpp>
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

visualization_msgs::Marker generate_marker(geometry_msgs::Point pt, double goal_x, double goal_y, int iter, double r, double g, double b)
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
      Marker_lane.color.r = r;
      Marker_lane.color.g = g;
      Marker_lane.color.b = b;
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
    ros::init(argc, argv, "test_frontier");
    ros::NodeHandle nh("");

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

    
    for(int i=0; i< 50; i++){
        for(int j=0; j<50; j++){
            grid_map::Index idx(i,j);
            gt_map.at("base", idx) = 0.5;
            // gt_map.at("SFC", idx) = 0.5;
        }
    }

    nav_msgs::OccupancyGrid occ_grid = converter.OccupancyGridConverter(gt_map);
    grid_map::Frontier ft;

    double range_max = 5.0; double range_min = 0.5; 
    double hangle_max = 180; double hangle_min = -180; double angle_resol = 5.0;
    double resol = 1.0;

    RayTracer::Raytracer raytracer(100.0, 100.0, 5, obstacles);

    RayTracer::Lidar_sensor lidar(range_max, range_min, hangle_max, hangle_min, angle_resol, 100.0, 100.0, resol, raytracer);

    int x_idx; int y_idx;
    nh.param<int>("x_idx", x_idx, 60);
    nh.param<int>("y_idx", y_idx, 60);
    cout << x_idx << ", " << y_idx << endl; 
    
    grid_map::Index idx(x_idx, y_idx);
    vector<vector<grid_map::Index> > frontier_vec = ft.wfd(gt_map, idx);
    grid_map::Position cur_pos; 
    grid_map::Position grid_cur_pos; 
    gt_map.getPosition(idx, cur_pos); grid_cur_pos = cur_pos;
    cur_pos(0,0) = cur_pos(0,0) + 50.0; 
    cur_pos(1,0) = cur_pos(1,0) + 50.0; 
    
    lidar.set_belief_map(gt_map);
    std::vector<Eigen::Vector2d > frontier_pos = lidar.frontier_detection(cur_pos);
    std::vector<std::vector<Eigen::Vector2d> > clustered_frontiers = lidar.frontier_clustering(frontier_pos);


    //SFC Start 
    cout << "SFC start" << endl; 
    grid_map::Size size_; size_ = gt_map.getSize();
    vector<Eigen::Vector2d> selected_ft; 
    selected_ft.push_back(frontier_pos.front());
    selected_ft.push_back(frontier_pos.back());
    lidar.set_selected_frontier(selected_ft);
    cout << "CHECKPOINT 0" << endl; 
    
    lidar.construct_SFC(cur_pos);

    cout << "CHECKPOINT 1" << endl; 
    std::pair<vec_E<Polyhedron<2>>, Eigen::Vector2d> corridor_pair = lidar.get_SFC();
    cout << "CHECKPOINT 2" << endl;
    std::vector<std::vector<Eigen::Vector2d> > jps_path = lidar.get_JPS_Path(grid_cur_pos);
    cout << "CHECKPOINT 3" << endl; 
    
    int num_t = 0;
    std::vector<visualization_msgs::Marker> jps_marker_vec; 
    for(int i=0; i<jps_path.size(); i++){
        for(int j=0; j<jps_path.at(i).size(); j++){
            num_t++;
            geometry_msgs::Point pt;
            pt.x = (jps_path.at(i).at(j))(0,0); pt.y = (jps_path.at(i).at(j))(1,0); pt.z = 0.0;
            visualization_msgs::Marker marker = generate_marker(pt, 0.0, 0.0, num_t, 0.0, 1.0, 0.0);
            jps_marker_vec.push_back(marker);
        }
    }

    //SFC End 

    std::random_device rd;

    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> dis(0, 20);

    vector<visualization_msgs::Marker> marker_vec;
    cout << "Size of vector" << frontier_vec.size() << " " << endl;
    int num = 0;
    cout << "Size of cluster" << clustered_frontiers.size() << " " << endl; 
    for(int i=0; i< clustered_frontiers.size(); i++){
        double r = dis(gen)/20.0; 
        for(int j=0; j<clustered_frontiers.at(i).size(); j++){
            num++;
            Eigen::Vector2d pos = clustered_frontiers.at(i).at(j);
            Eigen::Vector2d grid_pos = euc_to_gridref(pos, size_);
            // grid_map::Index idx = frontier_vec.at(i).at(j);
            // grid_map::Position pos;
            // gt_map.getPosition(idx, pos);

            geometry_msgs::Point pt; 
            pt.x = grid_pos(0,0); pt.y = grid_pos(1,0); pt.z = 0.0;
            // ROS_INFO("%d th Frontier point %f, %f in Index %d, %d", i*frontier_vec.size()+j, pt.x, pt.y, idx(0,0), idx(1,0));
            
            visualization_msgs::Marker marker = generate_marker(pt, 0.0, 0.0, num, r, 0.0, 1-r);
            marker_vec.push_back(marker);
        }
    }

    ros::Publisher pub = nh.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
    ros::Rate rate(30.0);
    ros::Publisher pub_occ = nh.advertise<nav_msgs::OccupancyGrid>("occu_grid", 1, true);
    ros::Publisher pub_vis_ft = nh.advertise<visualization_msgs::Marker>("frontier", 100, true);
    ros::Publisher pub_vis_jps = nh.advertise<visualization_msgs::Marker>("jps", 100, true);
    grid_map::Size size = gt_map.getSize();



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
        for(int i=0; i<jps_marker_vec.size(); i++){
            pub_vis_jps.publish(jps_marker_vec.at(i));
        }
        rate.sleep();    
    }

    return 0;
}