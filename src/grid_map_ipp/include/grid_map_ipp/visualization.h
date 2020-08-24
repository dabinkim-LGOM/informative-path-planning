#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <vector>
#include <string>
#include <ros/ros.h>

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
visualization_msgs::Marker generate_text_marker(geometry_msgs::Point pt, double goal_x, double goal_y, int iter, double r, double g, double b, string text)
  {
      visualization_msgs::Marker Marker_lane;
      Marker_lane.id = iter;
      Marker_lane.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
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
    
      Marker_lane.text = text; 
      pt.x = pt.x - goal_x;
      pt.y = pt.y - goal_y;
      // Marker_lane.header.stamp = ros::Time();
      Marker_lane.pose.position.x = pt.x;
      Marker_lane.pose.position.y = pt.y;
      Marker_lane.pose.position.z = pt.z;
      return Marker_lane;
 }

 visualization_msgs::MarkerArray generate_corridor_marker(std::vector<std::vector<double>> &box_vec)
{   
    visualization_msgs::MarkerArray marker_array; 
    marker_array.markers.clear();
    for(int i=0; i<box_vec.size(); i++){
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.type = visualization_msgs::Marker::CUBE;
        marker.color.a = 0.5;
        marker.color.r = 0;
        marker.color.g = 1;
        marker.color.b = 0;
        marker.pose.position.x = (box_vec[i][0] + box_vec[i][2]) / 2.0;
        marker.pose.position.y = (box_vec[i][1] + box_vec[i][3]) / 2.0;
        marker.pose.position.z = 0.0;
        marker.scale.x = box_vec[i][2] - box_vec[i][0];
        marker.scale.y = box_vec[i][3] - box_vec[i][1];
        marker.scale.z = 1.0;
        // marker.text =  to_string(corridor.t_start) + "-"+to_string(corridor.t_end);
        marker_array.markers.emplace_back(marker);
    }
    return marker_array;
}