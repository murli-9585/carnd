#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "PID.h"
#include <math.h>

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
    auto found_null = s.find("null");
    auto b1 = s.find_first_of("[");
    auto b2 = s.find_last_of("]");
    if (found_null != std::string::npos) {
        return "";
    }
    else if (b1 != std::string::npos && b2 != std::string::npos) {
        return s.substr(b1, b2 - b1 + 1);
    }
    return "";
}

int main()
{
    uWS::Hub h;

    // PID controller object for Steering value.
    PID pid, pid_throttle;

    // Iter1: Initialize PID variables.
    // First try with Twiddle on.
    //pid.Init(0.02, 0.5, 0.09); 

    // Iter 2:
    //  pid.Init(0.18, 0.03, 1.5);

    // Iter 3:
    //pid.Init(0.52, 0.50001, 1.0);

    //pid.Init(0.1232, 0.0001, 2.4564);
    // Last kind of successful with throttle at .4
    //pid.Init(0.1348, 0.0002302, 2.9564);

    // Updated with total error 14.xxx; n_steps 500.
    // pid.Init(0.14828, 0.00027624, 2.9564);

    // Updated after 3 or more rounds with correcting itself at critical;
    // Changing n_steps  to 1000 for a lap.
    //pid.Init(0.14828, 0.00027624, 2.95645);

    // Now changing to total known track ~2200 steps
    // Also incresing error steps to 200 with last known
    // good total error of 200
    //pid.Init(0.12828, 0.00027624, 3.54764);

    // Tried with constant throttle of 0.3, with N-steps 1500 (which seems
    // like 1 track.
    // pid.Init(0.141108, 0.000303864, 3.54764);

    // Ran this for 50 iterations with throttle at .3 These new are optimized.
    pid.Init(0.160741, 0.000303864, 3.6024, false, 1500);

    // Start with low values.
    // started with 0.33, 0.33, 0.33; It seems no good as
    // Overall Intergral of error does matter?? Its
    // more of short term so P and D seems realistic.
    //pid_throttle.Init(0.33, 0.0, 0.03, true, 100);

    pid_throttle.Init(0.3992, 0.0, 0.036, false, 500);

    h.onMessage([&pid, &pid_throttle](uWS::WebSocket<uWS::SERVER> ws,
                 char *data, size_t length, uWS::OpCode opCode) {
            // "42" at the start of the message means there's a websocket
            // message event.
            // The 4 signifies a websocket message
            // The 2 signifies a websocket event
            if (length && length > 2 && data[0] == '4' && data[1] == '2')
            {
            auto s = hasData(std::string(data).substr(0, length));
            if (s != "") {
            auto j = json::parse(s);
            std::string event = j[0].get<std::string>();
            if (event == "telemetry") {
            // j[1] is the data JSON object
            double cte = std::stod(j[1]["cte"].get<std::string>());
            //double speed = std::stod(j[1]["speed"].get<std::string>());
            //double angle = std::stod(j[1]["steering_angle"].get<std::string>());
            double avg_throttle = 0.27;
            double throttle_value;
            double steer_value;

            pid.UpdateError(cte);

            pid_throttle.UpdateError(cte);

            steer_value = pid.TotalError();
            // Steering value is [-1, 1]
            if (steer_value > 1.0) steer_value = 1.0;
            if (steer_value < -1.0) steer_value = -1.0;

            // For very minor changes keep the value close to average.
            throttle_value = avg_throttle - pid_throttle.TotalError();

            // XXXMM: Hack to put throttle in range.
            if (fabs(throttle_value) > avg_throttle/2) {
                        // Not to go above half the average speed; can be 1/3.
                        throttle_value = fmod(throttle_value, avg_throttle/2);
            }
            if (throttle_value > 0)
                        throttle_value = avg_throttle + throttle_value;
            else
                        throttle_value = 0;//avg_throttle - throttle_value;
            // DEBUG
            //std::cout << "-------------------------------------" <<std::endl;
            //std::cout << "CTE: " << cte << " Steering Value: " \
            //    << steer_value << std::endl;

            json msgJson;

            msgJson["steering_angle"] = steer_value;//deg2rad(steer_value);
            msgJson["throttle"] = throttle_value;
            auto msg = "42[\"steer\"," + msgJson.dump() + "]";
            std::cout << msg << " CTE: " << cte << std::endl;
            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
            } else {
                // Manual driving
                std::string msg = "42[\"manual\",{}]";
                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
            }
    });

    // We don't need this since we're not using HTTP but if it's
    // removed the program doesn't compile :-(
    h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req,
                char *data, size_t, size_t) {
            const std::string s = "<h1>Hello world!</h1>";
            if (req.getUrl().valueLength == 1)
            {
            res->end(s.data(), s.length());
            }
            else
            {
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
    if (h.listen(port))
    {
        std::cout << "Listening to port " << port << std::endl;
    }
    else
    {
        std::cerr << "Failed to listen to port" << std::endl;
        return -1;
    }
    h.run();
}
