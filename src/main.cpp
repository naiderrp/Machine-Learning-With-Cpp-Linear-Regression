#include <iostream>
#include "linear_regression.hpp"

int main() {
	std::vector<double> features = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	std::vector<double> targets = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

	// y = k * x + b
	auto [k, b] = linear_regression::train(features, targets);

	std::cout << "coef: " << k << "\nintercept: " << b << '\n';

	double x = 10'000;
	double y_pred = k * x + b;

	std::cout << y_pred;

	return 0;
}
