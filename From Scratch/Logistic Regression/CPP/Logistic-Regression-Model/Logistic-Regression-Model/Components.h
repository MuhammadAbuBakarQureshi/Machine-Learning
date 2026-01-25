#pragma once

#include <vector>

using namespace std;

vector<double> compute_z(vector<double> X, double w, double b);

vector<double> compute_model(vector<double> X, double w, double b);

double compute_cost(vector<double> X, vector<double> y, double w, double b);

pair<double, double> compute_gradient_descent_derivative(vector<double> X, vector<double> y, double w, double b);

pair<double, double> compute_gradient_descent(vector<double> X, vector<double> y, double w, double b, double alpha);

pair<double, double> train_model(vector<double> X, vector<double> y, double w, double b, double alpha, size_t epochs);

void threshold(vector<double>& y);