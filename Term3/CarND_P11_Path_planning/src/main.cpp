#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"
#include "map_waypoints.h"

using namespace std;
using json = nlohmann::json;
tk::spline spline_s;

double HZ = 0.02;
// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
        auto found_null = s.find("null");
        auto b1 = s.find_first_of("[");
        auto b2 = s.find_first_of("}");
        if (found_null != string::npos) {
                return "";
        } else if (b1 != string::npos && b2 != string::npos) {
                return s.substr(b1, b2 - b1 + 2);
        }
        return "";
}

int calculateLane(float lane_dist, double lane_width) {
        /*
            Lane representation from divider. Left most 0 and on.
           ||    |    |    |    |
           || 0  |  1 |  2 | 3  |
           ||    |    |    |    |
        */
        // This should probably be given by sensor fusion.
        // For now, assume its 3 lanes in one direction.
        // This is needed to avoid stalled cars and assuming that
        // shoulder lane is for passing.
        // This is going to fail if it has more than max_lanes.
        int max_lanes = 3;
        int lane =  (int) floor(lane_dist/lane_width);
        if (lane > max_lanes)
                return -1;
        else
                return lane;
}

vector<bool> processSensorFusion(vector<vector<double>> sensor_fusion,
    int prev_size, double car_s, int car_lane) {

        // All cars in same direction are observed.
        // Only cars which affect current car's lane and
        // speed are taken into consideration.
        bool blocking_front = false;
        bool blocking_left = false;
        bool blocking_right = false;
        double lane_width = 4.0; // Lane width.

        //find reference velocity to use
        for (int i=0; i < sensor_fusion.size(); i++) {
                // For every other car detected ..
                double vx = sensor_fusion[i][3];
                double vy = sensor_fusion[i][4];
                double ocar_speed = sqrt(vx*vx+vy*vy);
                double ocar_s = sensor_fusion[i][5];
                float ocar_d = sensor_fusion[i][6];

                int ocar_lane = calculateLane(ocar_d, lane_width);
                // Ignore any car which is away 30 meters.
                if ((ocar_s > (car_s + 30)) || (ocar_lane == -1))
                        continue;

                // if using prev points can project s value
                // outwards in time
                ocar_s +=((double)prev_size*HZ*ocar_speed);
                //car is in my lane
                if((ocar_lane == car_lane) && (ocar_s > car_s)) {
                        blocking_front = true;
                        /*  // Debug
                        cout << "FRONT BLOCKING: ocar_s: " << ocar_s;
                        cout << " car_s: " << car_s << " ocar_lane: " << ocar_lane
                            << " car_lane: " << car_lane << endl;
                        */
                }
                else if ((ocar_lane == car_lane+1) && (ocar_s > (car_s - 15)))
                        blocking_right = true;
                else if ((ocar_lane == car_lane-1) && (ocar_s > (car_s - 15)))
                        blocking_left = true;
        }
        vector<bool> next_states;
        next_states.push_back(blocking_front);
        next_states.push_back(blocking_left);
        next_states.push_back(blocking_right);
        return next_states;
}

void updateCarState(vector<bool> next_states, int& lane, double& ref_vel) {
            bool blocking_front = next_states[0];
            bool blocking_left = next_states[1];
            bool blocking_right = next_states[2];

            double vel_incr = .447; // .447 m/s = 1 MPH
            double vel_decr = .224;
            int to_lane = lane;
            // Change lanes when there is blocking in front of car.
            if (blocking_front) {
                    /*
                    std::cout <<std::boolalpha;
                    std::cout << "Left Blocking: " << blocking_left;
                    std::cout << " , Right Blocking: " << blocking_right << endl;
                    */
                    //switch into left or right lane accordingly when there
                    // are no closeby cars in those lanes
                    if (lane == 0 && !blocking_right)
                            to_lane = 1;
                    else if ((lane == 1) && ( !blocking_left))
                            to_lane = 0;
                    else if ((lane == 1) && ( !blocking_right))
                            to_lane = 2;
                    else if (lane == 2 && !blocking_left)
                            to_lane = 1;

                   // Lane changing, so increment velocity if needed.
                   if (lane != to_lane) {
                           lane = to_lane;
                           if (ref_vel < 49.5)
                                   ref_vel += vel_incr;
                        }
                   else { // In same lane, so reduce speed.
                           ref_vel -= vel_decr;
                   }
            }
            else if (ref_vel < 49.5) {
                    ref_vel += vel_incr;
            }
}

json getNextWaypoints(json& j, MapWaypoints& waypoints,
    int& lane, double& ref_vel) {

            int lookahead_wps_cnt = 50;
            // Main car's localization Data
            double car_x = j[1]["x"];
            double car_y = j[1]["y"];
            double car_s = j[1]["s"];
            double car_d = j[1]["d"];
            double car_yaw = j[1]["yaw"];
            double car_speed = j[1]["speed"];

            // previous path data
            auto previous_path_x = j[1]["previous_path_x"];
            auto previous_path_y = j[1]["previous_path_y"];
            // previous path s & d end values
            double end_path_s = j[1]["end_path_s"];
            double end_path_d = j[1]["end_path_d"];
            vector<vector<double>> sensor_fusion = j[1]["sensor_fusion"];

            int prev_size = previous_path_x.size();

            if (prev_size > 0) {
                    car_s = end_path_s;
            }
            // Process sensor fusion and determine if the car
            // is too close to front, or too close to left or
            // too close to right.
            vector<bool> next_states = processSensorFusion(sensor_fusion,
                prev_size, car_s, lane);

            // Based on the states, update velocity and lane of the car.
            updateCarState(next_states, lane, ref_vel);

            // Get next waypoints.
            // Ref: The code below is taken from Udacity walkthough.
            vector<double> ptsx;
            vector<double> ptsy;

            //reference x, y, yaw states either we will reference
            // the starting point as where the car is or at
            //the previous path's end point
            double ref_x = car_x;
            double ref_y = car_y;
            double ref_yaw = deg2rad(car_yaw);
            double ref_x_prev;
            double ref_y_prev;

            // if previous size is almost empty, use the car as starting reference
            if (prev_size < 2) {
                    //use two points that make the path tangent to the car
                    ref_x_prev = car_x - cos(car_yaw);
                    ref_y_prev = car_y - sin(car_yaw);
            }
            else {
                    // use the previous path's end point as starting reference
                    //redefine the reference state as previous path's end point.
                    // previous points help in determining/smooth-out next
                    // points. This avoids jerks.
                    ref_x = previous_path_x[prev_size-1];
                    ref_y = previous_path_y[prev_size-1];

                    ref_x_prev = previous_path_x[prev_size-2];
                    ref_y_prev = previous_path_y[prev_size-2];
                    ref_yaw = atan2(ref_y-ref_y_prev,ref_x-ref_x_prev);
            }
            ptsx.push_back(ref_x_prev);
            ptsx.push_back(ref_x);
            ptsy.push_back(ref_y_prev);
            ptsy.push_back(ref_y);

            //In frenet, add fwd distance evenly spaced points ahead of the
            // starting reference. Get points 2 & 3 times of required points.
            vector<double> next_wp0 = waypoints.getXY(
                car_s+lookahead_wps_cnt, (2+4*lane));
            vector<double> next_wp1 = waypoints.getXY(
                car_s+(2*lookahead_wps_cnt), (2+4*lane));
            vector<double> next_wp2 = waypoints.getXY(
                car_s+(3*lookahead_wps_cnt), (2+4*lane));

            ptsx.push_back(next_wp0[0]);
            ptsx.push_back(next_wp1[0]);
            ptsx.push_back(next_wp2[0]);

            ptsy.push_back(next_wp0[1]);
            ptsy.push_back(next_wp1[1]);
            ptsy.push_back(next_wp2[1]);

            // Shift points to avoid just horizantal or veritical points.
            // This is to avoid muliple y-values for given x or vice versa.
            for (int i = 0; i < ptsx.size(); i++){
                    //shift car reference angle to 0 degrees
                    double shift_x = ptsx[i]-ref_x;
                    double shift_y = ptsy[i]-ref_y;

                    // This is needed to avoid Horizantal or vertical lines, so
                    // when spline is queried, we dont get multiple y values for
                    // given x.
                    ptsx[i] = (shift_x * cos(0-ref_yaw)-shift_y*sin(0-ref_yaw));
                    ptsy[i] = (shift_x * sin(0-ref_yaw)+shift_y*cos(0-ref_yaw));
            }

            // set boundry; Not needed, done above.
            //spline_s.set_boundary(tk::spline::second_deriv, 4.0,
            //    tk::spline::first_deriv, 4.0, false);
            // set (x,y) points to the spline
            spline_s.set_points(ptsx, ptsy);

            // define the actual x, y points we will use for the planner
            vector<double> next_x_vals;
            vector<double> next_y_vals;

            // define the actual x, y points we will use for the planner
            for (int i = 0; i < previous_path_x.size(); i++){
                    next_x_vals.push_back(previous_path_x[i]);
                    next_y_vals.push_back(previous_path_y[i]);
            }

            //calculate how to break up spline points
            // so that we travel at our desired ref velocity
            double target_x = lookahead_wps_cnt;
            double target_y = spline_s(target_x);
            double target_dist = sqrt((target_x * target_x) +
                (target_y * target_y));

            double x_add_on = 0;

            // Fill up the rest of our path planner after filling it with
            // previous points, here we will always output
            // 50 points
            // HZ;50Hz loop in code.
            // 2.24 m/s increment over the course of 50 points to
            // achieve target velocity.
            for (int i = 1; i <= 50-previous_path_x.size(); i++){
                    double N = (target_dist/(HZ*ref_vel/2.24));
                    double x_point = x_add_on+(target_x)/N;
                    double y_point = spline_s(x_point);

                    x_add_on = x_point;

                    double x_ref = x_point;
                    double y_ref = y_point;

                    // rotate back to normal after rotating it earlier
                    x_point = (x_ref * cos(ref_yaw)-y_ref*sin(ref_yaw));
                    y_point = (x_ref * sin(ref_yaw)+y_ref*cos(ref_yaw));

                    x_point += ref_x;
                    y_point += ref_y;

                    next_x_vals.push_back(x_point);
                    next_y_vals.push_back(y_point);
            }

        json msgJson;
        msgJson["next_x"] = next_x_vals;
        msgJson["next_y"] = next_y_vals;
        return msgJson;
    }

int main() {
        uWS::Hub h;
        // Waypoint map to read from
        string map_file_ = "../data/highway_map.csv";
        MapWaypoints waypoints = MapWaypoints(map_file_);
        // The max s value before wrapping around the track back to 0
        double max_s = 6945.554;
        //start in lane 1;
        int lane = 1;
        //have a reference velocity to target
        double ref_vel = 0; //in mph

        h.onMessage([&waypoints, &ref_vel, &lane]
            (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
            uWS::OpCode opCode) {
                // "42" at the start of the message means there's a websocket
                //message event.
                // The 4 signifies a websocket message
                // The 2 signifies a websocket event
                //auto sdata = string(data).substr(0, length);
                //cout << sdata << endl;
                if (length && length > 2 && data[0] == '4' && data[1] == '2') {
                        auto s = hasData(data);
                        if (s != "") {
                                auto j = json::parse(s);
                                string event = j[0].get<string>();
                                if (event == "telemetry") {
                                        json msgJson = getNextWaypoints(j,
                                            waypoints, lane, ref_vel);
                                        auto msg = "42[\"control\","+ msgJson.dump()+"]";

                                        //this_thread::sleep_for(chrono::milliseconds(1000));
                                        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
                                }
                        } 
                        else {
                            // Manual driving
                            std::string msg = "42[\"manual\",{}]";
                            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
                        }
                    }
            });

        // We don't need this since we're not using HTTP but if it's removed the
        // program
        // doesn't compile :-(
        h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                size_t, size_t) {
            const std::string s = "<h1>Hello world!</h1>";
            if (req.getUrl().valueLength == 1) {
            res->end(s.data(), s.length());
            } else {
            // i guess this should be done more gracefully?
            res->end(nullptr, 0);
            }
            });

        h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
            std::cout << "Connected!!!" << std::endl;
            });

        h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                char *message, size_t length) {
            ws.close();
            std::cout << "Disconnected" << std::endl;
            });

        int port = 4567;
        if (h.listen(port)) {
                std::cout << "Listening to port " << port << std::endl;
        } else {
                std::cerr << "Failed to listen to port" << std::endl;
                return -1;
        }
        h.run();
}
