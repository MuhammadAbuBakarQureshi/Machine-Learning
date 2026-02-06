#include <iostream>
#include "../Components.h"

using namespace std;

double compute_cost(vector<double> X, vector<double> y,double w, double b) {

	vector<double> f_wb = compute_model(X, w, b);

	int m = size(X);

	double cost = 0.0;

	double epsilon = 1e-15;

	for (int i = 0; i < m; i++) {

		cost += (-y.at(i) * log(f_wb.at(i) + epsilon) - ( (1 - y.at(i)) * log(1 - f_wb.at(i) +epsilon ) ) );
	}
	
	cost /= m;

	return cost;
}