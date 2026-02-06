#include <iostream>
#include <vector>
#include "../Components.h"

using namespace std;

pair<double, double> train_model(vector<double> X, vector<double> y, double w, double b, double alpha, size_t epochs){

	for (size_t i = 0; i < epochs; i++) {

		pair<double, double> w_b = compute_gradient_descent(X, y, w, b, alpha);
		double cost = compute_cost(X, y, w, b);

		w = w_b.first;
		b = w_b.second;

		cout << "---Cost == " << cost << "			---w = " << w_b.first << "			---b = " << w_b.second << "			---Remaining = " << epochs - i - 1 << endl;
	}

	return make_pair(w, b);
}