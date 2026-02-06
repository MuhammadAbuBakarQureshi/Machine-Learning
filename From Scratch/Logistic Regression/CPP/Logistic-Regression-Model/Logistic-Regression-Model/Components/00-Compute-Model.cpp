#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

vector<double> compute_z(vector<double> X, double w, double b) {
		
	vector<double> f_wb(size(X));

	for (int i = 0; i < size(X); i++) {

		f_wb[i] = ( w * X[i] ) + b;
	}

	return f_wb;
}

vector<double> compute_model(vector<double> X, double w, double b) {

	vector<double> g_z(size(X));

	vector<double> z = compute_z(X, w, b);

	for (int i = 0; i < size(X); i++) {

		 double z_exp = exp(-z.at(i));

		 g_z.at(i) = 1 / (1 + z_exp);
	}
	
	return g_z;
}