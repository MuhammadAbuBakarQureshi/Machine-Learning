#pragma once

#include <iostream>

using namespace std;

// Train Model

	pair<double, double> train_model(double x[], double y[], double y_pred[], int data_size, double w, double b, double alpha, int epochs);


// Compute Model

	void compute_model(double x[], double y_pred[], int data_size, double w, double b);


// Gradient Descent 

	pair<double, double>  compute_gradient(double x[], double y[], int data_Size, double w, double b);

	pair<double, double> compute_gradient_descent(double x[], double y[], int data_size, double w, double b, double alpha);

// Compute Cost

	double compute_cost(double x[], double y[], int data_size, double w, double b);

// Print Data

	void print_data(double x[], int data_set_size);