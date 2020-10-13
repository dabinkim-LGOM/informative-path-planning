// #include "ros/ros.h"
// #include <cstdlib>
// #include <ctime>
// #include <queue>
// #include <map>
// #include <algorithm>
#include "grid_map_ipp/FFD.hpp"
// #include "nav_msgs/OccupancyGrid.h"
const double OCC_THRESHOLD = 0.85;
const double FREE_THRESHOLD = 0.15;
const int N_S = 8;
const int MIN_FOUND = 1;


using namespace std;

namespace grid_map{


    bool Ft_Detector::is_in_map(grid_map::Size map_size, grid_map::Index cur_index)
    {   
        // ROS_INFO("In In map");
        int map_size_x = map_size(0,0); int map_size_y = map_size(1,0);
        if(cur_index(0,0) < map_size_x && cur_index(0,0)>=0 && cur_index(1,0) <map_size_y && cur_index(1,0)>=0)
        {
            return true;
        }
        else{
            return false;
        }
    }

    void Ft_Detector::get_neighbours(grid_map::Index n_array[], grid_map::Index position)
    {   
        Eigen::Array2i n1(1,1);     Eigen::Array2i n4(0,1);     Eigen::Array2i n6(-1,1);
        Eigen::Array2i n2(1,0);     Eigen::Array2i n5(0,-1);    Eigen::Array2i n7(-1,0);
        Eigen::Array2i n3(1,-1);                                Eigen::Array2i n8(-1,-1);
        
        n_array[0] = position + n1;
        n_array[1] = position + n2;
        n_array[2] = position + n3;
        n_array[3] = position + n4;
        n_array[4] = position + n5;
        n_array[5] = position + n6;
        n_array[6] = position + n7;
        n_array[7] = position + n8;
        
    }

    // void get_neighbours2(int n_array[], int position, int map_width){
    //     n_array[0] = position - map_width - 1;
    //     n_array[1] = position - map_width;
    //     n_array[2] = position - map_width + 1;
    //     n_array[3] = position - 1;
    //     n_array[4] = position + 1;
    //     n_array[5] = position + map_width - 1;
    //     n_array[6] = position + map_width;
    //     n_array[7] = position + map_width + 1;
    // }

    bool Ft_Detector::is_frontier_point(const grid_map::GridMap& map, grid_map::Index point){
        // ROS_INFO("In Frontier Point");
        // The point under consideration must be known
        if( abs(map.at("base", point) - 0.5) < FREE_THRESHOLD) {
            return false;
        }
        if(map.at("base", point) > OCC_THRESHOLD)
            return false;
        grid_map::Size map_size = map.getSize();
        grid_map::Index locations[N_S]; 
        get_neighbours(locations, point);

        int found = 0;
        for(int i = 0; i < N_S; ++i) {
            // ROS_INFO("Location Index: %d, %d, %d", locations[i](0,0), locations[i](1,0), i);
            if(is_in_map(map_size, locations[i])) {
                // ROS_INFO("Outside of Is In Map");
                // None of the neighbours should be occupied space.		
                if(map.at("base",locations[i]) > OCC_THRESHOLD) {
                    return false;
                }
                //At least one of the neighbours is unknown, hence frontier point
                if( abs(map.at("base",locations[i]) - 0.5) < 0.2) {
                    found++;
                    //
                    if(found == MIN_FOUND) 
                        return true;
                }
                //}
            }
        }
        // ROS_INFO("Getting Out of Frontier Point");
        return false;

    }

    // bool is_frontier_point2(const nav_msgs::OccupancyGrid& map, int point, int map_size, int map_width){
    //     // The point under consideration must be known
    //     if(map.data[point] != -1)
    //     {
    //         return false;
    //     }
    //     //
    //     int locations[8];
    //     get_neighbours2(locations, point, map_width);
    //     for(int i = 0; i < 8; i++)
    //     {
    //         if(locations[i] < map_size && locations[i] >= 0)
    //         {
    //             //At least one of the neighbours is open and known space, hence frontier point //AANPASSING
    //             //if(map.data[locations[i]] < OCC_THRESHOLD && map.data[locations[i]] >= 0) {
    //             if(map.data[locations[i]] == -1)
    //             {
    //                 return true;
    //             }
    //         }
    //     }
    //     return false;
    // }


    float CrossProduct(float x0,float y0,float x1,float y1,float x2,float y2){
        return (x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0);
    }

    vector<Point> Ft_Detector::Sort_Polar( vector<Point> lr, Point pose ){
        //simple bubblesort
        bool swapped = 0;
        do
        {
            swapped = 0;
            for(unsigned int i=1; i<lr.size(); i++)
            {
                if( CrossProduct(pose.x,pose.y,lr[i-1].x,lr[i-1].y,lr[i].x,lr[i].y) > 0 )  //sorting clockwise
                {
                    Point swap = lr[i-1];
                    lr[i-1] = lr[i];
                    lr[i] = swap;
                    swapped = 1;
                }
            }
        }
        while(swapped);

        return lr;
    }


    //Bresenham's line algorithm
    Line Ft_Detector::Get_Line( Point prev, Point curr ){
        Line output;

        int x0=prev.x, y0=prev.y, x1=curr.x, y1=curr.y;

        int dx=x1-x0;
        int dy=y1-y0;

        float D = 2*dy - dx;

        Point newPoint;
        newPoint.x = x0;
        newPoint.y = y0;
        output.points.push_back(newPoint);

        int y=y0;

        for( int x=x0+1; x<=x1; x++)
        {
            if (D > 0)
            {
                y = y+1;
                Point newPoint;
                newPoint.x = x;
                newPoint.y = y;
                output.points.push_back(newPoint);
                D = D + (2*dy-2*dx);
            }
            else
            {
                Point newPoint;
                newPoint.x = x;
                newPoint.y = y;
                output.points.push_back(newPoint);
                D = D + (2*dy);
            }
        }
        return output;
    }


    /**
     * @brief Fast Frontier Detector main code. 
     * 
     * @param pose_idx : Current robot pose 
     * @param lr_idx : laser readings (Index of gridmap)
     * @param map : GridMap 
     * @return vector<vector<grid_map::Index> > 
     */
    vector<vector<grid_map::Index> > Ft_Detector::FFD( grid_map::Position pose, vector<grid_map::Index> lr_idx, const grid_map::GridMap& map){

        // polar sort readings according to robot position
        Point pose_mp; pose_mp.x = pose[0]; pose_mp.y = pose[1];

        vector<Point> lr;
        for(int i=0; i<lr_idx.size(); i++){
            Point cur_pt; cur_pt.x = lr_idx[i][0]; cur_pt.y = lr_idx[i][1];
            lr.push_back(cur_pt);
            std::cout << "x: " << cur_pt.x << "y: " << cur_pt.y << std::endl;
        }
        std::cout << "Size of laser scan: " << lr.size() << std::endl; 

        std::cout << "CHECKPT 0" << std::endl;
        vector<Point> sorted = Sort_Polar(lr,pose_mp);
        // get the contour from laser readings
        Point prev = sorted.back();
        sorted.pop_back();
        vector<Point> contour;

        std::cout << "CHECKPT 1" << std::endl;
        for(unsigned int i=0; i<sorted.size(); i++)
        {
            Line line = Get_Line(prev, sorted[i]);
            for(unsigned int j=0; j<line.points.size(); j++)
            {
                contour.push_back(line.points[j]);
            }
        }

        std::cout << "CHECKPT 2" << std::endl;
        // extract new frontiers from contour
        vector<Frontier> NewFrontiers;
        prev = contour.back(); //Point prev
        contour.pop_back();

        grid_map::Index prev_idx(prev.x, prev.y);
        if ( is_frontier_point(map, prev_idx) )
        {
            Frontier newFrontier;
            NewFrontiers.push_back(newFrontier);
        }
        int type1 = 0; int type2 = 0; int type3 = 0; int type4 = 0;
        for(unsigned int i=0; i<contour.size(); i++)
        {
            Point curr = contour[i];
            grid_map::Index curr_idx(curr.x, curr.y);
            if( !is_frontier_point(map, curr_idx) )
            {   type1++; 
                prev = curr;
            }
            // else if ( !(map.data[(curr.x + curr.y * map_width)] != -1) )    //curr is already visited
            else if ( std::abs(map.at("base", curr_idx) - 0.5) > 0.2)
            {   type2++; 
                prev = curr;
            }
            else if ( is_frontier_point(map, curr_idx)
                        &&  is_frontier_point(map, prev_idx) )
            {   type3++;
                NewFrontiers[NewFrontiers.size()-1].push_back(curr);
                prev = curr;
            }
            else
            {   type4++;
                Frontier newFrontier;
                newFrontier.push_back(curr);
                NewFrontiers.push_back(newFrontier);
                prev = curr;
            }
        }
        std::cout << "Type1: " << type1 << " Type2: " << type2 << " Type3: " << type3 << " Type4: " << type4 << std::endl; 
        std::cout << "# of Contour: " << contour.size()<<  std::endl;
        std::cout << "# of New Frontiers: " << NewFrontiers.size() << std::endl; 
        // maintainance of previously detected frontiers
        //Get active area
        int x_min=-90000,x_max=90000,y_min=-90000,y_max=90000;
        for(unsigned int i=0; i<lr.size(); i++)
        {
            x_max = min(x_max,lr[i].x);
            y_max = min(y_max,lr[i].y);
            x_min = max(x_min,lr[i].x);
            y_min = max(y_min,lr[i].y);
        }

        for(int x=x_min; x<=x_max; x++)
        {
            for(int y=y_min; y<=y_max; y++)
            {
                Point p;
                p.x = x;
                p.y = y;
                grid_map::Index p_idx(x, y);
                if( is_frontier_point(map, p_idx) )
                {
                    // split the current frontier into two partial frontiers
                    int Enables_f = -1;
                    int Enables_p = -1;
                    for(unsigned int i=0; i<frontiersDB.size(); i++)
                    {
                        for(unsigned int j=0; j<frontiersDB[i].size(); j++)
                        {
                            if(  frontiersDB[i][j].x == p.x && frontiersDB[i][j].y == p.y )
                            {
                                Enables_f = i;
                                Enables_p = j;
                            }
                        }//for j
                    }//for i

                    if(Enables_f == -1 || Enables_p == -1)
                        continue;

                    Frontier f1;
                    Frontier f2;
                    for(int i=0; i<=Enables_p; i++)
                    {
                        f1.push_back( frontiersDB[Enables_f][i] );
                    }
                    for(unsigned int i=Enables_p+1; i<frontiersDB[Enables_f].size(); i++)
                    {
                        f2.push_back( frontiersDB[Enables_f][i] );
                    }
                    frontiersDB.erase(frontiersDB.begin() + Enables_f);
                }//if p is a frontier

            } //for y
        }//for x

        std::cout << "CHECKPT 4" << std::endl;
        //Storing new detected frontiers
        grid_map::Size mapsize = map.getSize();
        int ActiveArea[ mapsize[0]][mapsize[1]];
        for(unsigned int i=0; i<frontiersDB.size(); i++)
        {
            for(unsigned int j=0; j<frontiersDB[i].size(); j++)
            {
                ActiveArea[frontiersDB[i][j].x][frontiersDB[i][j].y] = i;
            }
        }
        for(unsigned int i=0; i<NewFrontiers.size(); i++)
        {
            Frontier f = NewFrontiers[i];
            bool overlap = 0;
            for(unsigned int j=0; j<f.size(); j++)
            {
                if( ActiveArea[f[j].x][f[j].y] != 0 ) //overlap
                {
                    int exists = ActiveArea[f[j].x][f[j].y];
                    //merge f and exists
                    for(unsigned int merged=0; merged<f.size(); merged++)
                    {
                        frontiersDB[exists].push_back(f[merged]);
                    }
                    NewFrontiers[i].clear();
                    overlap = 1;
                    break;
                }
            }//for j
            if(overlap == 0)
            {
                frontiersDB.push_back(f);
            }
        }//for i
        std::cout << "CHECKPT 5" << std::endl;
        //remove empty frontier
        for(unsigned int i=0; i<frontiersDB.size(); i++)
        {
            if( frontiersDB[i].size() == 0 )
            {
                frontiersDB.erase(frontiersDB.begin() + i);
            }
        }


        vector<vector<grid_map::Index> > Result;
        //convert frontierDB to frontiers
        for(unsigned int i=0; i<frontiersDB.size(); i++){
            vector<grid_map::Index> NewFrontiers;
            vector<Point> ThisFrontier = frontiersDB[i];
            for(unsigned int j=0; j<ThisFrontier.size(); j++){
                grid_map::Index cur_idx(ThisFrontier[j].x, ThisFrontier[j].y);
                NewFrontiers.push_back(cur_idx );
            }
            Result.push_back(NewFrontiers);
        }

        std::cout << "CHECKPT 6" << std::endl;
        // vector<int> NewFrontiers2;
        // for(int x=0; x<5; x++){
        // for(int y=0; y<5; y++){
        // NewFrontiers2.push_back( x + (y * mapsize[0]) );
        // }
        // }

        // Result.push_back(NewFrontiers2);
        std::cout << "# of Frontier DB: " << frontiersDB.size() << std::endl; 
        std::cout << "Number of resulting Frontiers: " << Result.size() << std::endl; 
        return Result;

    }//end FFD










    // vector<vector<int> > Ft_Detector::FFD( Point pose,vector<Point> lr, const nav_msgs::OccupancyGrid& map, int map_height, int map_width){
    //     int map_size = map_height * map_width;

    //     // polar sort readings according to robot position
    //     vector<Point> sorted = Sort_Polar(lr,pose);
    //     // get the contour from laser readings
    //     Point prev = sorted.back();
    //     sorted.pop_back();
    //     vector<Point> contour;

    //     for(unsigned int i=0; i<sorted.size(); i++)
    //     {
    //         Line line = Get_Line(prev, sorted[i]);
    //         for(unsigned int j=0; j<line.points.size(); j++)
    //         {
    //             contour.push_back(line.points[j]);
    //         }
    //     }

    //     // extract new frontiers from contour
    //     vector<Frontier> NewFrontiers;
    //     prev = contour.back(); //Point prev
    //     contour.pop_back();


    //     if ( is_frontier_point2(map, (prev.x + prev.y * map_width)  ,map_size,map_width) )
    //     {
    //         Frontier newFrontier;
    //         NewFrontiers.push_back(newFrontier);
    //     }
    //     for(unsigned int i=0; i<contour.size(); i++)
    //     {
    //         Point curr = contour[i];
    //         if( !is_frontier_point2(map, (curr.x + curr.y * map_width)  ,map_size,map_width) )
    //         {
    //             prev = curr;
    //         }
    //         else if ( !(map.data[(curr.x + curr.y * map_width)] != -1) )    //curr is already visited
    //         {
    //             prev = curr;
    //         }
    //         else if ( is_frontier_point2(map, (curr.x + curr.y * map_width)  ,map_size,map_width)
    //                     &&  is_frontier_point2(map, (prev.x + prev.y * map_width)  ,map_size,map_width) )
    //         {
    //             NewFrontiers[NewFrontiers.size()-1].push_back(curr);
    //             prev = curr;
    //         }
    //         else
    //         {
    //             Frontier newFrontier;
    //             newFrontier.push_back(curr);
    //             NewFrontiers.push_back(newFrontier);
    //             prev = curr;
    //         }
    //     }


    //     // maintainance of previously detected frontiers
    //     //Get active area
    //     int x_min=-90000,x_max=90000,y_min=-90000,y_max=90000;
    //     for(unsigned int i=0; i<lr.size(); i++)
    //     {
    //         x_max = max(x_max,lr[i].x);
    //         y_max = max(y_max,lr[i].y);
    //         x_min = min(x_min,lr[i].x);
    //         y_min = min(y_min,lr[i].y);
    //     }

    //     for(int x=x_min; x<=x_max; x++)
    //     {
    //         for(int y=y_min; y<=y_max; y++)
    //         {
    //             Point p;
    //             p.x = x;
    //             p.y = y;
    //             if( is_frontier_point2(map, (p.x + p.y * map_width)  ,map_size,map_width) )
    //             {
    //                 // split the current frontier into two partial frontiers
    //                 int Enables_f = -1;
    //                 int Enables_p = -1;
    //                 for(unsigned int i=0; i<frontiersDB.size(); i++)
    //                 {
    //                     for(unsigned int j=0; j<frontiersDB[i].size(); j++)
    //                     {
    //                         if(  frontiersDB[i][j].x == p.x && frontiersDB[i][j].y == p.y )
    //                         {
    //                             Enables_f = i;
    //                             Enables_p = j;
    //                         }
    //                     }//for j
    //                 }//for i

    //                 if(Enables_f == -1 || Enables_p == -1)
    //                     continue;

    //                 Frontier f1;
    //                 Frontier f2;
    //                 for(int i=0; i<=Enables_p; i++)
    //                 {
    //                     f1.push_back( frontiersDB[Enables_f][i] );
    //                 }
    //                 for(unsigned int i=Enables_p+1; i<frontiersDB[Enables_f].size(); i++)
    //                 {
    //                     f2.push_back( frontiersDB[Enables_f][i] );
    //                 }
    //                 frontiersDB.erase(frontiersDB.begin() + Enables_f);
    //             }//if p is a frontier

    //         } //for y
    //     }//for x

    //     //Storing new detected frontiers
    //     int ActiveArea[ map.info.width][map.info.height];
    //     for(unsigned int i=0; i<frontiersDB.size(); i++)
    //     {
    //         for(unsigned int j=0; j<frontiersDB[i].size(); j++)
    //         {
    //             ActiveArea[frontiersDB[i][j].x][frontiersDB[i][j].y] = i;
    //         }
    //     }
    //     for(unsigned int i=0; i<NewFrontiers.size(); i++)
    //     {
    //         Frontier f = NewFrontiers[i];
    //         bool overlap = 0;
    //         for(unsigned int j=0; j<f.size(); j++)
    //         {
    //             if( ActiveArea[f[j].x][f[j].y] != 0 ) //overlap
    //             {
    //                 int exists = ActiveArea[f[j].x][f[j].y];
    //                 //merge f and exists
    //                 for(unsigned int merged=0; merged<f.size(); merged++)
    //                 {
    //                     frontiersDB[exists].push_back(f[merged]);
    //                 }
    //                 NewFrontiers[i].clear();
    //                 overlap = 1;
    //                 break;
    //             }
    //         }//for j
    //         if(overlap == 0)
    //         {
    //             frontiersDB.push_back(f);
    //         }
    //     }//for i

    //     //remove empty frontier
    //     for(unsigned int i=0; i<frontiersDB.size(); i++)
    //     {
    //         if( frontiersDB[i].size() == 0 )
    //         {
    //             frontiersDB.erase(frontiersDB.begin() + i);
    //         }
    //     }


    //     vector<vector<int> > Result;
    //     //convert frontierDB to frontiers
    //     for(unsigned int i=0; i<frontiersDB.size(); i++){
    //     vector<int> NewFrontiers;
    //     vector<Point> ThisFrontier = frontiersDB[i];
    //     for(unsigned int j=0; j<ThisFrontier.size(); j++){
    //     NewFrontiers.push_back(   ThisFrontier[j].x + (ThisFrontier[j].y * map.info.width) );
    //     }
    //     Result.push_back(NewFrontiers);
    //     }


    //     vector<int> NewFrontiers2;
    //     for(int x=0; x<5; x++){
    //     for(int y=0; y<5; y++){
    //     NewFrontiers2.push_back( x + (y * map.info.width) );
    //     }
    //     }

    //     Result.push_back(NewFrontiers2);

    //     return Result;

    // }//end FFD


}
