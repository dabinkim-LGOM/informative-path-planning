#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;

namespace clustering{
    class Point{
    private:
        int pointId, clusterId;
        int dimensions;
        vector<double> values;

    public:
    
        Point(int id, Eigen::Vector2d& pt){
            dimensions = 2;
            pointId = id;
            // stringstream is(line);
            values.emplace_back(pt(0,0));
            values.emplace_back(pt(1,0));
        
            clusterId = 0; //Initially not assigned to any cluster
        }

        int getDimensions(){
            return dimensions;
        }

        int getCluster(){
            return clusterId;
        }

        int getID(){
            return pointId;
        }

        void setCluster(int val){
            clusterId = val;
        }
        vector<double> getVal(){
            return values;
        }
        // double getVal(int pos){
        //     return values[pos];
        // }

    };

    class Cluster{
    private:
        int clusterId;
        vector<double> centroid;
        vector<Point> points;
    public:
        Cluster(int clusterId, Point& pt){
            this->clusterId = clusterId;
            for(int i=0; i<pt.getDimensions(); i++){
                this->centroid=pt.getVal();
            }
            this->addPoint(pt);
        }
        void addPoint(Point& p){
            p.setCluster(this->clusterId);
            points.push_back(p);
        }
        
        bool removePoint(int pointId){
            int size = points.size();

            for(int i = 0; i < size; i++)
            {
                if(points[i].getID() == pointId)
                {
                    points.erase(points.begin() + i);
                    return true;
                }
            }
            return false;
        }
        int getId(){
            return clusterId;
        }

        Point getPoint(int pos){
            return points[pos];
        }

        int getSize(){
            return points.size();
        }

        vector<double> getCentroidByPos() {
            return centroid;
        }

        void setCentroidByPos(vector<double> val){
            this->centroid = val;
        }
    };

    class KMeans{
        private:
        int K, iters, dimensions, total_points;
        vector<Cluster> clusters;
        
        int getNearestClusterId(Point& point){
            double sum = 0.0, min_dist;
            int NearestClusterId;

            // for(int i = 0; i < dimensions; i++)
            // {
                // sum += pow(clusters[0].getCentroidByPos(i) - point.getVal(i), 2.0);
            sum = ((clusters[0].getCentroidByPos())[0] - (point.getVal())[0]) * ((clusters[0].getCentroidByPos())[0] - (point.getVal())[0])
                + ((clusters[0].getCentroidByPos())[1] - (point.getVal())[1]) * ((clusters[0].getCentroidByPos())[1] - (point.getVal())[1]);
            // }

            min_dist = sqrt(sum);
            NearestClusterId = clusters[0].getId();

            for(int i = 1; i < K; i++)
            {
                double dist;
                // sum = 0.0;
                // for(int j = 0; j < dimensions; j++)
                // {
                    // sum += pow(clusters[i].getCentroidByPos(j) - point.getVal(j), 2.0);
                sum = ((clusters[i].getCentroidByPos())[0] - (point.getVal())[0]) * ((clusters[i].getCentroidByPos())[0] - (point.getVal())[0])
                    +((clusters[i].getCentroidByPos())[1] - (point.getVal())[1]) * ((clusters[i].getCentroidByPos())[1] - (point.getVal())[1]);
                // }
                dist = sqrt(sum);

                if(dist < min_dist)
                {
                    min_dist = dist;
                    NearestClusterId = clusters[i].getId();
                }
            }
            return NearestClusterId;
        }

        public:
        KMeans(int K, int iterations){
            this->K = K;
            this->iters = iterations;
        }
        void run(vector<Point>& all_points){

            total_points = all_points.size();
            dimensions = all_points[0].getDimensions();


            //Initializing Clusters
            vector<int> used_pointIds;

            for(int i=1; i<=K; i++)
            {
                while(true)
                {
                    int index = rand() % total_points;

                    if(find(used_pointIds.begin(), used_pointIds.end(), index) == used_pointIds.end())
                    {
                        used_pointIds.push_back(index);
                        all_points[index].setCluster(i);    
                        Cluster cluster(i, all_points[index]);
                        clusters.push_back(cluster);
                        break;
                    }
                }
            }
            std::cout << "Initial Cluster number = " << K << std::endl; 
            std::cout<<"Clusters initialized = "<<clusters.size()<<endl<<endl;


            std::cout<<"Running K-Means Clustering.."<<endl;

            int iter = 1;
            while(true)
            {
                std::cout<<"Iter - "<<iter<<"/"<<iters<<endl;
                bool done = true;

                // Add all points to their nearest cluster
                for(int i = 0; i < total_points; i++)
                {
                    int currentClusterId = all_points[i].getCluster();
                    int nearestClusterId = getNearestClusterId(all_points[i]);

                    if(currentClusterId != nearestClusterId)
                    {
                        if(currentClusterId != 0){
                            for(int j=0; j<K; j++){
                                if(clusters[j].getId() == currentClusterId){
                                    clusters[j].removePoint(all_points[i].getID());
                                }
                            }
                        }

                        for(int j=0; j<K; j++){
                            if(clusters[j].getId() == nearestClusterId){
                                clusters[j].addPoint(all_points[i]);
                            }
                        }
                        all_points[i].setCluster(nearestClusterId);
                        done = false;
                        break;
                    }
                }

                // Recalculating the center of each cluster
                for(int i = 0; i < K; i++)
                {
                    int ClusterSize = clusters[i].getSize();

                    vector<double> sum; 
                    for(int j = 0; j < dimensions; j++)
                    {   
                        // Eigen::Vector2d sum(0.0, 0.0);
                        double sum_tmp = 0;
                        if(ClusterSize > 0)
                        {
                            for(int p = 0; p < ClusterSize; p++)
                            {    sum_tmp += (clusters[i].getPoint(p).getVal())[j];
                            }
                            sum_tmp = sum_tmp/ClusterSize; 
                            // sum[1] = sum[1]/ClusterSize; 
                            sum.emplace_back(sum_tmp);
                        }
                    }
                    clusters[i].setCentroidByPos(sum);

                }

                if(done || iter >= iters)
                {
                    std::cout << "Clustering completed in iteration : " <<iter<<endl<<endl;
                    break;
                }
                iter++;
            }


        //Print pointIds in each cluster
        for(int i=0; i<K; i++){
            std::cout<<"Points in cluster "<<clusters[i].getId()<<" : ";
            for(int j=0; j<clusters[i].getSize(); j++){
                std::cout<<clusters[i].getPoint(j).getID()<<" ";
            }
            std::cout<<endl<<endl;
        }
        std::cout<<"========================"<<endl<<endl;

        //Write cluster centers to file
        // ofstream outfile;
        // outfile.open("clusters.txt");
        // if(outfile.is_open()){
        //     for(int i=0; i<K; i++){
        //         cout<<"Cluster "<<clusters[i].getId()<<" centroid : ";
        //         for(int j=0; j<dimensions; j++){
        //             cout<<clusters[i].getCentroidByPos(j)<<" ";     //Output to console
        //             outfile<<clusters[i].getCentroidByPos(j)<<" ";  //Output to file
        //         }
        //         cout<<endl;
        //         outfile<<endl;
        //     }
        //     outfile.close();
        // }
        // else{
        //     cout<<"Error: Unable to write to clusters.txt";
        // }

    }
        vector<Cluster> get_cluster(){ return clusters;}
    };
}