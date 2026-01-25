#include <iostream>

using namespace std;

pair<double, double>  compute_gradient(double x[], double y[], int data_Size, double w, double b) {

    double dw = 0;
    double db = 0;

    for (int i = 0; i < data_Size; i++) {

        double fwb = (w * x[i]) + b;
        double dw_i = (fwb - y[i]) * x[i];
        double db_i = (fwb - y[i]);

        dw += dw_i;
        db += db_i;
    }

    dw /= data_Size;
    db /= data_Size;

    return make_pair(dw, db);
}


pair<double, double> compute_gradient_descent(double x[], double y[], int data_size, double w, double b, double alpha) {

    pair<double, double> dw_db = compute_gradient(x, y, data_size, w, b);

    w = w - (alpha * dw_db.first);
    b = b - (alpha * dw_db.second);

    return make_pair(w, b);
}
