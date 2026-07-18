#pragma once

#include <cstddef>
#include <limits>
#include <vector>

namespace patience {

struct TracePoint {
    std::size_t cost = 0;
    bool result_changed = false;
    double average_topk_distance = 0.0;
    std::vector<std::size_t> labels;
};

struct SearchTrace {
    std::vector<TracePoint> points;
    std::size_t total_cost = 0;
    std::vector<std::size_t> final_labels;
};

struct StopResult {
    std::size_t cost = 0;
    bool stopped_early = false;
    double score = std::numeric_limits<double>::infinity();
    std::vector<std::size_t> labels;
};

class HardPatience {
public:
    explicit HardPatience(std::size_t tau);
    StopResult evaluate(const SearchTrace& trace) const;
    std::size_t tau() const noexcept { return tau_; }

private:
    std::size_t tau_;
};

class AdamPatience {
public:
    AdamPatience(double lambda_exponent, double beta1 = 0.9, double beta2 = 0.99);
    StopResult evaluate(const SearchTrace& trace) const;

    double lambda_exponent() const noexcept { return lambda_exponent_; }
    double threshold() const noexcept { return threshold_; }
    double beta1() const noexcept { return beta1_; }
    double beta2() const noexcept { return beta2_; }

private:
    double score_after_zeros(
        std::size_t zeros,
        std::size_t step,
        double momentum,
        double scale) const;

    double lambda_exponent_;
    double threshold_;
    double beta1_;
    double beta2_;
};

}  // namespace patience
