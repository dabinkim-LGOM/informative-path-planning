#ifndef JPS_H
#define JPS_H

#include <queue>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <map>

namespace Planner{
    class Node {
// Variables used here are constantly accessed and checked; leaving public for now.
    public:
        /** \brief x coordinate */
        int x_;
        /** \brief y coordinate */
        int y_;
        /** \brief Node id */
        int id_;
        /** \brief Node's parent's id */
        int pid_;
        /** \brief cost to reach this node */
        double cost_;
        /** \brief heuristic cost to reach the goal */
        double h_cost_;

        /**
        * @brief Constructor for Node class
        * @param x X value
        * @param y Y value
        * @param cost Cost to get to this node
        * @param h_cost Heuritic cost of this node
        * @param id Node's id
        * @param pid Node's parent's id
        */
        Node(int x = 0, int y = 0, double cost = 0, double h_cost = 0, int id = 0, int pid = 0) {
            this->x_ = x;
            this->y_ = y;
            this->cost_ = cost;
            this->h_cost_ = h_cost;
            this->id_ = id;
            this->pid_ = pid;
        }

        /**
        * @brief Prints the values of the variables in the node
        * @return void
        */
        void PrintStatus() {
            std::cout << "--------------" << std::endl
                      << "Node          :" << std::endl
                      << "x             : " << x_ << std::endl
                      << "y             : " << y_ << std::endl
                      << "Cost          : " << cost_ << std::endl
                      << "Heuristic cost: " << h_cost_ << std::endl
                      << "Id            : " << id_ << std::endl
                      << "Pid           : " << pid_ << std::endl
                      << "--------------" << std::endl;
        }

        /**
        * @brief Overloading operator + for Node class
        * @param p node
        * @return Node with current node's and input node p's values added
        */
        Node operator+(Node p) {
            Node tmp;
            tmp.x_ = this->x_ + p.x_;
            tmp.y_ = this->y_ + p.y_;
            tmp.cost_ = this->cost_ + p.cost_;
            return tmp;
        }

        /**
        * @brief Overloading operator - for Node class
        * @param p node
        * @return Node with current node's and input node p's values subtracted
        */
        Node operator-(Node p) {
            Node tmp;
            tmp.x_ = this->x_ - p.x_;
            tmp.y_ = this->y_ - p.y_;
            return tmp;
        }

        /**
        * @brief Overloading operator = for Node class
        * @param p node
        * @return void
        */
        void operator=(Node p) {
            this->x_ = p.x_;
            this->y_ = p.y_;
            this->cost_ = p.cost_;
            this->h_cost_ = p.h_cost_;
            this->id_ = p.id_;
            this->pid_ = p.pid_;
        }

        /**
        * @brief Overloading operator == for Node class
        * @param p node
        * @return bool whether current node equals input node
        */
        bool operator==(Node p) {
            if (this->x_ == p.x_ && this->y_ == p.y_) return true;
            return false;
        }

        /**
        * @brief Overloading operator != for Node class
        * @param p node
        * @return bool whether current node is not equal to input node
        */
        bool operator!=(Node p) {
            if (this->x_ != p.x_ || this->y_ != p.y_) return true;
            return false;
        }
    };

    /**
* @brief Struct created to encapsulate function compare cost between 2 nodes. Used in with multiple algorithms and classes
*/
    struct compare_cost {

        /**
        * @brief Compare cost between 2 nodes
        * @param p1 Node 1
        * @param p2 Node 2
        * @return Returns whether cost to get to node 1 is greater than the cost to get to node 2
        */
        bool operator()(Node &p1, Node &p2) {
            // Can modify this to allow tie breaks based on heuristic cost if required
            if (p1.cost_ + p1.h_cost_ > p2.cost_ + p2.h_cost_) return true;
            else if (p1.cost_ + p1.h_cost_ == p2.cost_ + p2.h_cost_ && p1.h_cost_ >= p2.h_cost_) return true;
            return false;
        }
    };

    std::vector<Node> GetMotion() {
        Node down(0, 1, 1, 0, 0, 0);
        Node up(0, -1, 1, 0, 0, 0);
        Node left(-1, 0, 1, 0, 0, 0);
        Node right(1, 0, 1, 0, 0, 0);
        std::vector<Node> v;
        v.push_back(down);
        v.push_back(up);
        v.push_back(left);
        v.push_back(right);
        // NOTE: Add diagonal movements for A* and D* only after the heuristics in the
        // algorithms have been modified. Refer to README.md. The heuristics currently
        // implemented are based on Manhattan distance and dwill not account for
        //diagonal/ any other motions
        return v;
    }

    class JumpPointSearch{
        private:
        std::priority_queue<Node, std::vector<Node>, compare_cost> open_list_;
        std::vector<std::vector<int>> grid;
        std::map<int, Node> closed_list_;
        std::vector<Node> skeleton_path_;
        std::unordered_set<int> pruned;
        Node start_, goal_;
        int dimx, dimy;

        public:
        std::vector<Node> jump_point_search(std::vector<std::vector<int>> &grid, Node start_in, Node goal_in);

        void InsertionSort(std::vector<Node> &v);

        bool has_forced_neighbours(Node &new_point, Node &next_point, Node &motion);

        Node jump(Node &new_point, Node &motion, int id);

        void backtracking(const Node &last_point);

    };
}