#include <iostream>
#include <vector>
#include "../Components.h"

using namespace std;

pair<double, double> compute_gradient_descent_derivative(vector<double> X, vector<double> y, double w, double b){

	vector<double> f_wb = compute_model(X, w, b);

	size_t m = size(X);

	double dw = 0, db = 0;

	for (size_t i = 0; i < m; i++) {

		dw += (f_wb.at(i) - y.at(i)) * X.at(i);
		db += f_wb.at(i) - y.at(i);
	}

	dw /= m;
	db /= m;

	return make_pair(dw, db);
}

pair<double, double> compute_gradient_descent(vector<double> X, vector<double> y, double w, double b, double alpha) {

	pair<double, double> dw_db = compute_gradient_descent_derivative(X, y, w, b);

	w -= alpha * dw_db.first;
	b -= alpha * dw_db.second;

	return make_pair(w, b);
}