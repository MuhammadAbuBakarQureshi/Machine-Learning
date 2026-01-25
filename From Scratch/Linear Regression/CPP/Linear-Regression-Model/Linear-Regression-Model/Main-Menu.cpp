#include <iostream>
#include "Model.h"

using namespace std;

int main()
{

    double x_train[10] = { 0, 5, 10, 15, 20, 25, 30, 35, 40, 45 };

    double y_train[10] = { 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 };

    double y_pred[10];

    int data_size = size(x_train);

    double w = 0, b = 0, alpha = 0.001;

    int epochs = 1000;

    pair<double, double> w_b = train_model(x_train, y_train, y_pred, data_size, w, b, alpha, epochs);

    // make predictions

    cout << "\n\n\n" << endl;

    cout << "Original : ";

    print_data(y_train, data_size);

    cout << "\n\n" << endl;
    
    cout << "Predicted : ";

    print_data(y_pred, data_size);

    return 0;
}