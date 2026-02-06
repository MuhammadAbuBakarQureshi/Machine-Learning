

void compute_model(double x[], double y_pred[], int data_size, double w, double b) {

    for (int i = 0; i < data_size; i++) {

        y_pred[i] = (w * x[i]) + b;
    }

}