#include <math.h>

double compute_cost(double x[], double y[], int data_size, double w, double b) {

    double total_cost = 0;
 
    for (int i = 0; i < data_size; i++) {

        double fwb = (w * x[i]) + b;
        fwb = fwb - y[i];

        total_cost += fwb;
    }

    total_cost = pow(total_cost, 2);

    total_cost /= (2 * data_size);

    return total_cost;
}