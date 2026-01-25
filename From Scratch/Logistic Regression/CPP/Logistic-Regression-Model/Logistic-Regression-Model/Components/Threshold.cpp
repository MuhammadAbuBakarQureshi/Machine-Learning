#include <iostream>
#include <vector>

using namespace std;

void threshold(vector<double>& y) {

	size_t m = size(y);

	for (size_t i = 0; i < m; i++) {

		if (y.at(i) >= 0.5) {
			
			y.at(i) = 1;
		}
		else {

			y.at(i) = 0;
		}
	}
}