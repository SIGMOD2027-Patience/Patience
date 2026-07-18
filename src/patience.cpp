#include "patience.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace patience {
namespace {

void validate_trace(const SearchTrace& trace) {
    std::size_t previous = 0;
    for (const auto& point : trace.points) {
        if (point.cost == 0 || point.cost < previous || point.cost > trace.total_cost) {
            throw std::invalid_argument("trace costs must be ordered and within total_cost");
        }
        if (!std::isfinite(point.average_topk_distance) ||
            point.average_topk_distance < 0.0) {
            throw std::invalid_argument("top-k distances must be finite and nonnegative");
        }
        previous = point.cost;
    }
}

StopResult natural_end(const SearchTrace& trace) {
    return StopResult{trace.total_cost, false,
                      std::numeric_limits<double>::infinity(), trace.final_labels};
}

}  // namespace

HardPatience::HardPatience(std::size_t tau) : tau_(tau) {}

StopResult HardPatience::evaluate(const SearchTrace& trace) const {
    validate_trace(trace);
    if (trace.points.empty()) {
        return natural_end(trace);
    }

    const TracePoint* current = &trace.points.front();
    std::size_t last_change = current->cost;
    for (std::size_t i = 1; i < trace.points.size(); ++i) {
        const TracePoint& next = trace.points[i];
        if (next.result_changed) {
            if (next.cost - last_change > tau_) {
                return StopResult{last_change + tau_, true,
                                  std::numeric_limits<double>::infinity(), current->labels};
            }
            last_change = next.cost;
            current = &next;
        }
    }

    if (trace.total_cost - last_change >= tau_ && last_change + tau_ < trace.total_cost) {
        return StopResult{last_change + tau_, true,
                          std::numeric_limits<double>::infinity(), current->labels};
    }
    return natural_end(trace);
}

AdamPatience::AdamPatience(double lambda_exponent, double beta1, double beta2)
    : lambda_exponent_(lambda_exponent),
      threshold_(std::isinf(lambda_exponent) ? 0.0 : std::pow(10.0, -lambda_exponent)),
      beta1_(beta1),
      beta2_(beta2) {
    if ((!std::isfinite(lambda_exponent_) && !std::isinf(lambda_exponent_)) ||
        lambda_exponent_ < 0.0) {
        throw std::invalid_argument("lambda exponent must be nonnegative or infinity");
    }
    if (!(beta1_ > 0.0 && beta1_ < 1.0) || !(beta2_ > 0.0 && beta2_ < 1.0)) {
        throw std::invalid_argument("beta values must be in (0, 1)");
    }
}

double AdamPatience::score_after_zeros(
    std::size_t zeros, std::size_t step, double momentum, double scale) const {
    if (scale <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const auto n = static_cast<double>(step + zeros);
    const double denominator = 1.0 - std::pow(beta1_, n);
    if (denominator <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return (momentum / scale) * std::pow(beta1_ / beta2_, static_cast<double>(zeros)) /
           denominator;
}

StopResult AdamPatience::evaluate(const SearchTrace& trace) const {
    validate_trace(trace);
    if (trace.points.empty() || threshold_ == 0.0) {
        return natural_end(trace);
    }

    const TracePoint* current = &trace.points.front();
    std::size_t previous_cost = current->cost;
    double previous_theta = std::log(std::max(current->average_topk_distance, 1e-30));
    double momentum = 0.0;
    double scale = 0.0;
    std::size_t step = 0;

    auto stop_in_gap = [&](std::size_t gap) -> std::size_t {
        if (gap == 0 || scale <= 0.0 ||
            score_after_zeros(gap, step, momentum, scale) >= threshold_) {
            return 0;
        }
        std::size_t lo = 1;
        std::size_t hi = gap;
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (score_after_zeros(mid, step, momentum, scale) < threshold_) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    };

    for (std::size_t i = 1; i < trace.points.size(); ++i) {
        const TracePoint& point = trace.points[i];
        const std::size_t gap = point.cost > previous_cost ? point.cost - previous_cost - 1 : 0;
        const std::size_t zero_stop = stop_in_gap(gap);
        if (zero_stop != 0) {
            return StopResult{previous_cost + zero_stop, true,
                              score_after_zeros(zero_stop, step, momentum, scale),
                              current->labels};
        }
        if (gap != 0) {
            step += gap;
            momentum *= std::pow(beta1_, static_cast<double>(gap));
            scale *= std::pow(beta2_, static_cast<double>(gap));
        }

        const double theta = std::log(std::max(point.average_topk_distance, 1e-30));
        const double progress = std::max(previous_theta - theta, 0.0);
        previous_theta = theta;
        ++step;
        momentum = beta1_ * momentum + (1.0 - beta1_) * progress;
        scale = std::max(beta2_ * scale, std::abs(progress));
        current = &point;
        previous_cost = point.cost;

        const double denominator = 1.0 - std::pow(beta1_, static_cast<double>(step));
        const double score = scale > 0.0 ? momentum / (denominator * scale)
                                         : std::numeric_limits<double>::infinity();
        if (score < threshold_) {
            return StopResult{point.cost, true, score, point.labels};
        }
    }

    const std::size_t zero_stop = stop_in_gap(trace.total_cost - previous_cost);
    if (zero_stop != 0) {
        return StopResult{previous_cost + zero_stop, true,
                          score_after_zeros(zero_stop, step, momentum, scale), current->labels};
    }
    return natural_end(trace);
}

}  // namespace patience
