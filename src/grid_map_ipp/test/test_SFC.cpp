#include "grid_map_core/GridMap.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ipp/wavefront_frontier_detection.hpp>
#include <grid_map_ipp/ObstacleGridConverter.hpp>
#include <grid_map_ipp/grid_map_ipp.hpp>
#include <grid_map_ipp/visualization.h>
#include <decomp_ros_utils/data_ros_utils.h>
#include <string>
// #include <decomp_ros_utils/polyhedron_array_display.h>
// #include <grid_map_ipp/util.hpp>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
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
    nh.param<int>("x_idx", x_idx, 80);
    nh.param<int>("y_idx", y_idx, 80);
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
    std::vector<Eigen::Vector2d> clustered_frontiers = lidar.frontier_clustering(frontier_pos);


    //SFC Start 
    grid_map::Size size_; size_ = gt_map.getSize();
    vector<Eigen::Vector2d> selected_ft; 
    selected_ft.push_back(frontier_pos.front());
    selected_ft.push_back(frontier_pos.at(5));
    selected_ft.push_back(frontier_pos.back());
    lidar.set_selected_frontier(selected_ft);
    
    lidar.construct_SFC(cur_pos);

    std::vector<std::pair<vec_E<Polyhedron<2>>, Eigen::Vector2d> > corridor_pair_vec = lidar.get_SFC();
    std::vector<std::vector<Eigen::Vector2d> > jps_path = lidar.get_JPS_Path(cur_pos);
    cout << "JPS_PATH" << endl; 
    for(int i=0; i<jps_path.size(); i++){
        cout << i <<"-th JPS Path" << endl; 
        for(int j=0; j<jps_path.at(i).size(); j++){
            cout << jps_path.at(i).at(j).transpose() << endl; 
        }
    }


    //SFC_jwp Start
    lidar.construct_SFC_jwp(cur_pos);
    std::vector<std::vector<std::vector<double>> > sfc_jwp_vec = lidar.get_SFC_jwp();

    std::vector<visualization_msgs::MarkerArray> sfc_vis_vec; 
    cout << "SIZE: " << sfc_jwp_vec.size() << endl;
    for(int i=0; i<sfc_jwp_vec.size(); i++){
        // std::vector<std::vector<double> > cur_box_vec;

        cout << ""<< endl;
        for(int j=0; j<sfc_jwp_vec.at(i).size(); j++){
            auto cur_vec = sfc_jwp_vec.at(i).at(j);
            for(int k=0; k<cur_vec.size(); k++){
                cout << "i: " << i << " j: " << j << " k: " << k << " val: " << cur_vec.at(k) << endl;
            }
        }
        visualization_msgs::MarkerArray sfc_vis = generate_corridor_marker(sfc_jwp_vec.at(i)); 
        sfc_vis_vec.push_back(sfc_vis);
    }



    cout << "CHECK -1" << endl; 
    int num_t = 0;
    std::vector<visualization_msgs::Marker> jps_marker_vec; 
    for(int i=0; i<jps_path.size(); i++){
        for(int j=0; j<jps_path.at(i).size(); j++){
            num_t++;    
            geometry_msgs::Point pt;
            pt.x = (jps_path.at(i).at(j))(0,0); pt.y = (jps_path.at(i).at(j))(1,0); pt.z = 0.0;
            std::string st_index = std::to_string(pt.x) + " " + std::to_string(pt.y);
            // cout << st_index << endl; 
            visualization_msgs::Marker marker = generate_marker(pt, 0.0, 0.0, num_t, 0.0, 0.0, 0.0);
            jps_marker_vec.push_back(marker);
        }
    }

    // cout << "CHECK0" << endl; 
    //SFC End 

    std::random_device rd;

    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> dis(0, 20);

    vector<visualization_msgs::Marker> marker_vec;
    // cout << "Size of vector" << frontier_vec.size() << " " << endl;
    int num = 0;
    // cout << "Size of cluster" << clustered_frontiers.size() << " " << endl; 
    // for(int i=0; i< clustered_frontiers.size(); i++){
    //     double r = dis(gen)/20.0; 
    //     for(int j=0; j<clustered_frontiers.at(i).size(); j++){
    //         num++;
    //         Eigen::Vector2d pos = clustered_frontiers.at(i).at(j);
    //         Eigen::Vector2d grid_pos = euc_to_gridref(pos, size_);
    //         // grid_map::Index idx = frontier_vec.at(i).at(j);
    //         // grid_map::Position pos;
    //         // gt_map.getPosition(idx, pos);

    //         geometry_msgs::Point pt; 
    //         pt.x = grid_pos(0,0); pt.y = grid_pos(1,0); pt.z = 0.0;
    //         // ROS_INFO("%d th Frontier point %f, %f in Index %d, %d", i*frontier_vec.size()+j, pt.x, pt.y, idx(0,0), idx(1,0));
            
    //         visualization_msgs::Marker marker = generate_marker(pt, 0.0, 0.0, num, r, 0.0, 1-r);
    //         marker_vec.push_back(marker);
    //     }
    // }

    cout << "CHECK1" << endl; 

    ros::Publisher pub = nh.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
    ros::Rate rate(30.0);
    ros::Publisher pub_occ = nh.advertise<nav_msgs::OccupancyGrid>("occu_grid", 1, true);
    ros::Publisher pub_vis_ft = nh.advertise<visualization_msgs::Marker>("frontier", 100, true);
    ros::Publisher pub_vis_jps = nh.advertise<visualization_msgs::Marker>("jps", 100, true);
    ros::Publisher pub_vis_sfc = nh.advertise<visualization_msgs::MarkerArray>("sfc", 100, true);
    // ros::Publisher es_pub = nh.advertise<decomp_ros_msgs::EllipsoidArray>("ellipsoid_array", 1, true);
    // ros::Publisher poly_pub = nh.advertise<decomp_ros_msgs::PolyhedronArray>("polyhedron_array", 1, true);
    grid_map::Size size = gt_map.getSize();




    
    // cout << "CHECK2" << endl; 
    // cout << jps_path.size() << endl; 
    // cout << "CORRIDOR size: " << corridor_pair_vec.size() << endl; 
    // for(size_t j = 0; j < jps_path.size(); j++) {
    //     cout << "HEY: " << jps_path.at(j).size() << endl; 
    //     for(size_t i=0; i<jps_path.at(j).size() -1; i++){
            
    //         const auto pt_inside = ( (jps_path.at(j))[i] + (jps_path.at(j))[i+1] ) / 2;
    //         cout << "CHECK2.0" << endl; 
    //         auto corr = ((corridor_pair_vec.at(j)).first);
    //         cout << "CHECK2.1" << endl; 
            
    //         auto polys = corr.at(i).hyperplanes();

    //         cout << "CHCEK3" << endl;

    //         LinearConstraint2D cs(pt_inside, polys);
    //         printf("i: %zu\n", i);
    //         std::cout << "A: " << cs.A() << std::endl;
    //         std::cout << "b: " << cs.b() << std::endl;

    //         std::cout << "point: " << (jps_path.at(j))[i].transpose();
    //         if(cs.inside((jps_path.at(j))[i]))
    //         std::cout << " is inside!" << std::endl;
    //         else
    //         std::cout << " is outside!" << std::endl;
    //         std::cout << "point: " << (jps_path.at(j))[i+1].transpose();
    //         if(cs.inside((jps_path.at(j))[i+1]))
    //         std::cout << " is inside!" << std::endl ;
    //         else
    //         std::cout << " is outside!" << std::endl;
    //     }
    // }


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

        for(int i=0; i<sfc_vis_vec.size(); i++){
            pub_vis_sfc.publish(sfc_vis_vec.at(0));
        }
        // poly_pub.publish(poly_msg);
        rate.sleep();    
    }

    return 0;
}