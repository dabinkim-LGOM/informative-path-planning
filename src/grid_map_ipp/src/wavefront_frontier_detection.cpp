// #include "grid_map_ipp/wavefront_frontier_detection.hpp"

// using namespace std;

// namespace grid_map{

//     class Frontier{
            

//         bool is_in_map(grid_map::Size map_size, grid_map::Index cur_index)
//     {
//         int map_size_x = map_size(0,0); int map_size_y = map_size(0,1);
//         if(cur_index(0,0) < map_size_x && cur_index(0,0)>=0 && cur_index(0,1) <map_size_y && cur_index(0,1)>=0)
//         {
//             return true;
//         }
//         else{
//             return false;
//         }
//     }

//     vector<vector<grid_map::Index> > wfd(const grid_map::GridMap& map_, grid_map::Index pose)
//     {	
        
//         vector<vector<grid_map::Index> > frontiers;
//         // Cell state list for map/frontier open/closed
//         // int map_size = map_height * map_width;
//         grid_map::Size map_size = map_.getSize();
//         map<grid_map::Index, int, less<grid_map::Index> > cell_states;
        
//         queue<grid_map::Index> q_m;	
//         q_m.push(pose);
//         cell_states[pose] = MAP_OPEN_LIST;
//         grid_map::Index adj_vector[N_S];
//         grid_map::Index v_neighbours[N_S];
//         //
//         //ROS_INFO("wfd 1");
//         while(!q_m.empty()) 
//         {
//             //ROS_INFO("wfd 2");
//             grid_map::Index cur_pos = q_m.front();
//             q_m.pop();
//             //ROS_INFO("cur_pos: %d, cell_state: %d",cur_pos, cell_states[cur_pos]);
//             // Skip if map_close_list
//             if(cell_states[cur_pos] == MAP_CLOSE_LIST)
//                 continue;
//             if(is_frontier_point(map_, cur_pos)) {
//                 queue<grid_map::Index> q_f;
//                 vector<grid_map::Index> new_frontier;
//                 q_f.push(cur_pos);
//                 cell_states[cur_pos] = FRONTIER_OPEN_LIST;
//                 // Second BFS
//                 while(!q_f.empty()) {
//                     //ROS_INFO("wfd 3");
//                     //ROS_INFO("Size: %d", q_f.size());
//                     grid_map::Index n_cell = q_f.front();
//                     q_f.pop();
//                     //
//                     if(cell_states[n_cell] == MAP_CLOSE_LIST || cell_states[n_cell] == FRONTIER_CLOSE_LIST)
//                         continue;
//                     //
//                     if(is_frontier_point(map_, n_cell)) {
//                         //ROS_INFO("adding %d to frontiers", n_cell);
//                         new_frontier.push_back(n_cell);
//                         get_neighbours(adj_vector, cur_pos);			
//                         //
//                         //ROS_INFO("wfd 3.5");
//                         for(int i = 0; i < N_S; i++) {
//                             if(is_in_map(map_size, adj_vector[i])) {
//                                 if(cell_states[adj_vector[i]] != FRONTIER_OPEN_LIST && 
//                                     cell_states[adj_vector[i]] != FRONTIER_CLOSE_LIST && 
//                                     cell_states[adj_vector[i]] != MAP_CLOSE_LIST) {
//                                     //ROS_INFO("wfd 4");
//                                     if(map_.at("base",adj_vector[i]) != 100) {
//                                         q_f.push(adj_vector[i]);
//                                         cell_states[adj_vector[i]] = FRONTIER_OPEN_LIST;
//                                     }
//                                 }
//                             }
//                         }
//                     }
//                     cell_states[n_cell] = FRONTIER_CLOSE_LIST;
//                 }
//                 if(new_frontier.size() > 2)
//                     frontiers.push_back(new_frontier);
                
//                 //ROS_INFO("WFD 4.5");
//                 for(unsigned int i = 0; i < new_frontier.size(); i++) {
//                     cell_states[new_frontier[i]] = MAP_CLOSE_LIST;
//                     //ROS_INFO("WFD 5");
//                 }
//             }
//             //
//             get_neighbours(adj_vector, cur_pos);

//             for (int i = 0; i < N_S; ++i) {
//                 //ROS_INFO("wfd 6");
//                 if( is_in_map(map_size, adj_vector[i])) {
//                     if(cell_states[adj_vector[i]] != MAP_OPEN_LIST &&  cell_states[adj_vector[i]] != MAP_CLOSE_LIST) {
//                         get_neighbours(v_neighbours, adj_vector[i]);
//                         bool map_open_neighbor = false;
//                         for(int j = 0; j < N_S; j++) {
//                             if(is_in_map(map_size, v_neighbours[j])) {
//                                 if(map_.at("base", v_neighbours[j]) < OCC_THRESHOLD && map_.at("base", v_neighbours[j]) >= 0) { //>= 0 AANPASSING
//                                     map_open_neighbor = true;
//                                     break;
//                                 }
//                             }
//                         }
//                         if(map_open_neighbor) {
//                             q_m.push(adj_vector[i]);
//                             cell_states[adj_vector[i]] = MAP_OPEN_LIST;
//                         }
//                     }
//                 }
//             }
//             //ROS_INFO("wfd 7");
//             cell_states[cur_pos] = MAP_CLOSE_LIST;
//             //ROS_INFO("wfd 7.1");
//         }
//         // ROS_INFO("wfd 8");
//         return frontiers;
//     }


//     void get_neighbours(grid_map::Index n_array[], grid_map::Index position)
//     {
//         n_array[0] = position + (1,1);
//         n_array[1] = position + (1,0);
//         n_array[2] = position + (1,-1);
//         n_array[3] = position + (0,1);
//         n_array[4] = position + (0,-1);
//         n_array[5] = position + (-1,1);
//         n_array[6] = position + (-1,0);
//         n_array[7] = position + (-1,-1);
        
//     }

//     void get_big_neighbours(int n_array[], int position, int map_width)
//     {
//         n_array[0] = position - map_width - 1;
//         n_array[1] = position - map_width; 
//         n_array[2] = position - map_width + 1; 
//         n_array[3] = position - 1;
//         n_array[4] = position + 1;
//         n_array[5] = position + map_width - 1;
//         n_array[6] = position + map_width;
//         n_array[7] = position + map_width + 1;

//         n_array[8] = position - (map_width * 2) - 2;
//         n_array[9] = position - (map_width * 2) - 1; 
//         n_array[10] = position - (map_width * 2); 
//         n_array[11] = position - (map_width * 2) + 1;
//         n_array[12] = position - (map_width * 2) + 2;
//         n_array[13] = position - 2;
//         n_array[14] = position + 2;
//         n_array[15] = position + (map_width * 2) - 2;
//         n_array[16] = position + (map_width * 2) - 1; 
//         n_array[17] = position + (map_width * 2); 
//         n_array[18] = position + (map_width * 2) + 1;
//         n_array[19] = position + (map_width * 2) + 2;
//         n_array[20] = position + (map_width) + 2;
//         n_array[21] = position + (map_width) - 2;
//         n_array[22] = position - (map_width) + 2;
//         n_array[23] = position - (map_width) - 2;
//     }



//     bool is_frontier_point(const grid_map::GridMap& map, grid_map::Index point)
//     {
//         // The point under consideration must be known
//         if(map.at("base", point) == -1) {
//             return false;
//         }
//         grid_map::Size map_size = map.getSize();
//         grid_map::Index locations[N_S]; 
//         get_neighbours(locations, point);
//         int found = 0;
//         for(int i = 0; i < N_S; i++) {
//             if(is_in_map(map_size, locations[i])) {
//                 // None of the neighbours should be occupied space.		
//                 if(map.at("base",locations[i]) > OCC_THRESHOLD) {
//                     return false;
//                 }
//                 //At least one of the neighbours is open and known space, hence frontier point
//                 if(map.at("base",locations[i]) < FREE_THRESHOLD) {
//                     found++;
//                     //
//                     if(found == MIN_FOUND) 
//                         return true;
//                 }
//                 //}
//             }
//         }
//         return false;
//     }

// };
// }