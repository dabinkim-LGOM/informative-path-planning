#include <SFC/JPS.h>


Planner::Node Planner::JumpPointSearch::jump(Node &new_point, Node &motion, int id) {
    Node next_point = new_point + motion;
    next_point.id_ = dimy * next_point.x_ + next_point.y_;
    next_point.pid_ = id;
    next_point.h_cost_ = abs(next_point.x_ - goal_.x_) + abs(next_point.y_ - goal_.y_);
    if (next_point.x_ < 0 || next_point.y_ < 0 || next_point.x_ >= dimx || next_point.y_ >= dimy ||
        grid[next_point.x_][next_point.y_] != 0) {
        return new_point;
        // return Node(-1,-1,-1,-1,-1,-1);
    }
    if (pruned.find(next_point.id_) != pruned.end()) pruned.insert(next_point.id_);
    if (next_point == goal_) return next_point;
    bool fn = false;
    fn = has_forced_neighbours(new_point, next_point, motion);
    if (fn) {
        // std::cout << "Forced neighbours found"<<std::endl;
        return next_point;
    } else {
        Node jump_node = jump(next_point, motion, id);
        // Prevent over shoot
        if (jump_node.cost_ != -1 &&
            jump_node.cost_ + jump_node.h_cost_ <= next_point.cost_ + next_point.h_cost_)
            return jump_node;
        else return next_point;
    }
}

bool Planner::JumpPointSearch::has_forced_neighbours(Node &new_point, Node &next_point, Node &motion) {
    int cn1x = new_point.x_ + motion.y_;
    int cn1y = new_point.y_ + motion.x_;

    int cn2x = new_point.x_ - motion.y_;
    int cn2y = new_point.y_ - motion.x_;

    int nn1x = next_point.x_ + motion.y_;
    int nn1y = next_point.y_ + motion.x_;

    int nn2x = next_point.x_ - motion.y_;
    int nn2y = next_point.y_ - motion.x_;

    bool a = !(cn1x < 0 || cn1y < 0 || cn1x >= dimx || cn1y >= dimy || grid[cn1x][cn1y] == 1);
    bool b = !(nn1x < 0 || nn1y < 0 || nn1x >= dimx || nn1y >= dimy || grid[nn1x][nn1y] == 1);
    if (a != b) return true;

    a = !(cn2x < 0 || cn2y < 0 || cn2x >= dimx || cn2y >= dimy || grid[cn2x][cn2y] == 1);
    b = !(nn2x < 0 || nn2y < 0 || nn2x >= dimx || nn2y >= dimy || grid[nn2x][nn2y] == 1);
    if (a != b) return true;

    return false;
}

void Planner::JumpPointSearch::backtracking(const Node &last_point) {
    skeleton_path_.clear();
    Node current = last_point;
    while (current.id_ != current.pid_) {
        skeleton_path_.emplace_back(current);
        current = closed_list_.find(current.pid_)->second;
    }
    skeleton_path_.emplace_back(current);
    std::reverse(skeleton_path_.begin(), skeleton_path_.end());
}

void Planner::JumpPointSearch::InsertionSort(std::vector<Node>& v){
    int nV = v.size();
    int i, j;
    Node key;
    for (i = 1; i < nV; i++) {
        key = v[i];
        j = i-1;
        while (j >= 0 && (v[j].cost_ + v[j].h_cost_ > key.cost_+key.h_cost_)){
            v[j+1] = v[j];
            j--;
        }
        v[j+1] = key;
    }
}

std::vector<Planner::Node>
Planner::JumpPointSearch::jump_point_search(std::vector<std::vector<int>> &grid, Node start_in, Node goal_in) {
    this->grid = grid;
    start_ = start_in;
    goal_ = goal_in;
    dimx = grid.size();
    dimy = grid[0].size();
    // Get possible motions
    std::vector<Node> motion = GetMotion();
    open_list_.push(start_);

    // Main loop
    Node temp;
    while (!open_list_.empty()) {
        Node current = open_list_.top();
        open_list_.pop();
        current.id_ = current.x_ * dimy + current.y_;
        if (current.x_ == goal_.x_ && current.y_ == goal_.y_) {
            closed_list_.insert(std::make_pair(current.id_, current));
            grid[current.x_][current.y_] = 2;
            backtracking(current);
            return skeleton_path_;
        }
        grid[current.x_][current.y_] = 2; // Point opened
        int current_cost = current.cost_;
        for (auto it = motion.begin(); it != motion.end(); ++it) {
            Node new_point;
            new_point = current + *it;
            new_point.id_ = dimy * new_point.x_ + new_point.y_;
            new_point.pid_ = current.id_;
            new_point.h_cost_ = abs(new_point.x_ - goal_.x_) + abs(new_point.y_ - goal_.y_);
            if (new_point == goal_) {
                open_list_.push(new_point);
                break;
            }
            if (new_point.x_ < 0 || new_point.y_ < 0 || new_point.x_ >= dimx || new_point.y_ >= dimy)
                continue; // Check boundaries
            if (grid[new_point.x_][new_point.y_] != 0) {
                continue; //obstacle or visited
            }

            Node jump_point = jump(new_point, *it, current.id_);
            if (jump_point.id_ != -1) {
                open_list_.push(jump_point);
                if (jump_point.x_ == goal_.x_ && jump_point.y_ == goal_.y_) {
                    closed_list_.insert(std::make_pair(current.id_, current));
                    closed_list_.insert(std::make_pair(jump_point.id_, jump_point));
                    grid[jump_point.x_][jump_point.y_] = 2;
                    backtracking(jump_point);
                    return skeleton_path_;
                }
            }
            open_list_.push(new_point);
        }
        closed_list_.insert(std::make_pair(current.id_, current));
    }
    skeleton_path_.clear();
    Node no_path_node(-1, -1, -1, -1, -1, -1);
    skeleton_path_.emplace_back(no_path_node);
    return skeleton_path_;
}
