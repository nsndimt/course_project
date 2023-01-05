#include <math.h>
#include <stdio.h>
#include <time.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

const double pi = 3.1415926535897932384626433832795;

double rad(double d) {
    return d * pi / 180.0;
}

double geo_distance(double lon1, double lat1, double lon2, double lat2, int test_cnt) {

    double a, b, s;
    double *distance = new double[test_cnt]{};
    #pragma omp parallel for 
    for (int i = 0; i < test_cnt; i++) {
        double radLat1 = rad(lat1);
        double radLat2 = rad(lat2);
        a = radLat1 - radLat2;
        b = rad(lon1) - rad(lon2);
        s = pow(sin(a/2),2) + cos(radLat1) * cos(radLat2) * pow(sin(b/2),2);
        distance[i] = 2 * asin(sqrt(s)) * 6378 * 1000;
    }

    return distance[0];
}

PYBIND11_MODULE (multi_thread, m) {
    m.def("geo_distance", &geo_distance, "Compute geography distance between two places.");
}
