#ifndef MAP_WAYPOINTS_H
#define MAP_WAYPOINTS_H
#include <vector>
#include <math.h>

using std::vector;
using namespace std;

class MapWaypoints {
public:
        MapWaypoints(string config_file);
        vector<double> getXY(double s, double di);
private:
        vector<double> map_waypoints_x;
        vector<double> map_waypoints_y;
        vector<double> map_waypoints_s;
        vector<double> map_waypoints_dx;
        vector<double> map_waypoints_dy;

};

#endif // define map_waypoints.h
