#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>

#include "map_waypoints.h"
using namespace std;

MapWaypoints::MapWaypoints(const string config_file) {

        ifstream conf_file(config_file.c_str(), ifstream::in);
        assert (conf_file.good() == true);
        string line;
        while (getline(conf_file, line)) {
                istringstream iss(line);
                double x;
                double y;
                float s;
                float d_x;
                float d_y;
                iss >> x;
                iss >> y;
                iss >> s;
                iss >> d_x;
                iss >> d_y;
                MapWaypoints::map_waypoints_x.push_back(x);
                MapWaypoints::map_waypoints_y.push_back(y);
                MapWaypoints::map_waypoints_s.push_back(s);
                MapWaypoints::map_waypoints_dx.push_back(d_x);
                MapWaypoints::map_waypoints_dy.push_back(d_y);
        }
}

// Transform from Frenet s,d coordinates to Cartesian x,y of given map.
vector<double>
MapWaypoints::getXY(double s, double d) {
            int prev_wp = -1;

            while(s > MapWaypoints::map_waypoints_s[prev_wp+1] &&
                (prev_wp < (int)(MapWaypoints::map_waypoints_s.size()-1) ))
                    prev_wp++;

            int wp2 = (prev_wp+1)%MapWaypoints::map_waypoints_x.size();

            double heading = atan2((MapWaypoints::map_waypoints_y[wp2]-
                MapWaypoints::map_waypoints_y[prev_wp]),
                (MapWaypoints::map_waypoints_x[wp2]-
                MapWaypoints::map_waypoints_x[prev_wp]));
            // the x,y,s along the segment
            double seg_s = (s-MapWaypoints::map_waypoints_s[prev_wp]);

            double seg_x = MapWaypoints::map_waypoints_x[prev_wp]+
                seg_s*cos(heading);
            double seg_y = MapWaypoints::map_waypoints_y[prev_wp]+
                seg_s*sin(heading);

            double perp_heading = heading-M_PI/2;

            double x = seg_x + d*cos(perp_heading);
            double y = seg_y + d*sin(perp_heading);

            return {x,y};

}
