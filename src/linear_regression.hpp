#pragma once
#include <vector>

struct linear_regression {
	static auto gradMSE(const std::vector<double>& x, const std::vector<double>& y, double w1, double w0) {
		std::vector<double> y_predicted(x.size());
		for (int i = 0; i < y_predicted.size(); ++i) y_predicted[i] = w1 * x[i] + w0;

		double grad_w0 = 0.0;
		double grad_w1 = 0.0;

		for (int i = 0; i < y_predicted.size(); ++i) {
			grad_w0 += (y_predicted[i] - y[i]);
			grad_w1 += (y_predicted[i] - y[i]) * x[i];
		}

		grad_w0 *= (2.0 / x.size());
		grad_w1 *= (2.0 / x.size());

		return std::make_pair(grad_w0, grad_w1);
	}

	static auto train(const std::vector<double>& x, const std::vector<double>& y) {
		double learning_rate = x.size() >= 5 ? 0.01 : 0.1;
		double eps = 0.0001;

		double w0 = 0.0;
		double w1 = 0.0;

		double next_w0 = w0;
		double next_w1 = w1;

		for (int n = 0; n < 1000; ++n) {
			double current_w0 = next_w0;
			double current_w1 = next_w1;

			auto [grad_current_w0, grad_current_w1] = gradMSE(x, y, current_w1, current_w0);

			next_w0 = current_w0 - learning_rate * grad_current_w0;
			next_w1 = current_w1 - learning_rate * grad_current_w1;

			if ((std::abs(next_w0 - current_w0) <= eps) && (std::abs(next_w1 - current_w1) <= eps))
				break;
		}
		return std::make_pair(next_w1, next_w0);
	}
};
