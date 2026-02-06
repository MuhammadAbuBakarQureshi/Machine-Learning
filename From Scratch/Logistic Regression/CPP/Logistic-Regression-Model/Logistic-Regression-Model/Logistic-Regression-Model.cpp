#include <iostream>
#include <vector>
#include "Components.h"

using namespace std;

void print_vector(vector<double> vec) {

	for (int i = 0; i < size(vec); i++) {

		cout << vec.at(i) << "  ";
	}cout << endl;
}

void eligible_age(double w, double b) {

	cout << "\n\n1. Enter age." << endl
		 << "2. Exit." << endl;

	int option;

	cout << "\nEnter your option : ";

	cin >> option;

	vector<double> age(1, 0);
	vector<double> y_pred(1, 0);

	switch (option) {

	case 1: 

		cout << "\n\n\tEnter your age : ";
		cin >> age.at(0);
		
		y_pred = compute_model(age, w, b);
		threshold(y_pred);

		if (y_pred.at(0) == 1) {

			cout << "You can participate in this program" << endl;
		}
		else {
			
			cout << "You cannot participate in this program" << endl;
		}

		eligible_age(w, b);
		break;

	case 2:

		cout << "Exiting" << endl;
		break;

	default:

		cout << "Entered wrong value" << endl;
		eligible_age(w, b);
		break;
	}


}

int main() {

	vector<double> X {20, 18, 10, 2, 13, 24, 11, 32, 54, 35, 67, 4, 16, 17};
	
	vector<double> y {1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0 };

	double w = 0, b = 0, alpha = 0.001;

	size_t epochs = 150000;

	pair<double, double> w_b = train_model(X, y, w, b, alpha, epochs);

	vector<double> y_pred = compute_model(X, w_b.first, w_b.second);

	threshold(y_pred);
	
	cout << "\n\n\n\n" << endl;

	cout << "Original : ";

	print_vector(y);

	cout << "\n" << endl;

	cout << "Predicted : ";

	print_vector(y_pred);

	eligible_age(w_b.first, w_b.second);

	return 0;
}