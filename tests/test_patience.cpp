#include "patience.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// The top-k improves quickly at cost 2 and then remains unchanged.  Adam's
// decaying progress score recognizes the stable tail before a fixed tau=30.
patience::SearchTrace stable_tail_trace() {
    patience::SearchTrace trace;
    trace.total_cost = 50;
    trace.final_labels = {2, 3};
    trace.points.push_back({1, true, 10.0, {1, 2}});
    trace.points.push_back({2, true, 5.0, {2, 3}});
    for (std::size_t cost = 3; cost <= trace.total_cost; ++cost) {
        trace.points.push_back({cost, false, 5.0, {2, 3}});
    }
    return trace;
}

void test_hard_patience() {
    const auto trace = stable_tail_trace();
    const auto result = patience::HardPatience(30).evaluate(trace);
    require(result.stopped_early, "hard patience should stop in the stable tail");
    require(result.cost == 32, "hard patience should wait exactly tau=30");
    require(result.labels == trace.final_labels, "hard patience must preserve top-k labels");

    const auto natural = patience::HardPatience(60).evaluate(trace);
    require(!natural.stopped_early, "tau beyond trace end must not stop early");
    require(natural.cost == trace.total_cost, "natural cost mismatch");
}

void test_adam_patience() {
    const auto trace = stable_tail_trace();
    const auto disabled = patience::AdamPatience(
        std::numeric_limits<double>::infinity()).evaluate(trace);
    require(!disabled.stopped_early, "infinite Lambda must disable early stopping");
    require(disabled.cost == trace.total_cost, "disabled Adam must reach natural end");

    const auto hard = patience::HardPatience(30).evaluate(trace);
    const auto adam = patience::AdamPatience(1.0).evaluate(trace);
    require(adam.stopped_early, "Adam-patience should detect the stable tail");
    require(adam.cost < hard.cost, "Adam-patience should beat fixed hard patience");
    require(adam.cost < trace.total_cost, "Adam-patience should beat efSearch");
    require(adam.labels == trace.final_labels, "Adam-patience must preserve top-k labels");

    std::cout << "Adam-patience cost: " << adam.cost << '\n'
              << "Hard Patience cost: " << hard.cost << '\n'
              << "efSearch cost: " << trace.total_cost << '\n';
}

void test_validation() {
    bool threw = false;
    try {
        patience::AdamPatience invalid(2.0, 1.0, 0.99);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "invalid beta must throw");
}

}  // namespace

int main() {
    test_hard_patience();
    test_adam_patience();
    test_validation();
    std::cout << "All patience tests passed\n";
    return 0;
}
