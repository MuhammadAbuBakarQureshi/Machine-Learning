#include <iostream>
#include "Model.h"

using namespace std;

pair<double, double> train_model(double x[], double y[], double y_pred[], int data_size, double w, double b, double alpha, int epochs) {

    for (int i = 0; i < epochs; i++) {

        compute_model(x, y_pred, data_size, w, b);
        double cost = compute_cost(x, y, data_size, w, b);
        pair<double, double> w_b = compute_gradient_descent(x, y, data_size, w, b, alpha);

        w = w_b.first;
        b = w_b.second;

        cout << "\t-------- COST = " << cost << " -------- w = " << w << " -------- b = " << b << " -------- Epochs Remaining = " << epochs -  (i + 1) << endl;
    }

    return make_pair(w, b);
}
