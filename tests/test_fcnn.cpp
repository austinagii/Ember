#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "fcnn.h"

TEST(FCNN, Constructor) {
    hyper::FCNN model({5, 10, 10, 1});
    std::vector<std::vector<std::size_t>> expectedWeightsShape = {{10, 5}, {10, 10}, {1, 10}};

    EXPECT_EQ(model.get_weights().size(), expectedWeightsShape.size());
    for (size_t i = 0; i < model.get_weights().size(); ++i) {
        auto layerWeights = model.get_weights().at(i);
        EXPECT_EQ(layerWeights.shape(), expectedWeightsShape[i]);

        // Check that all weights are between 0 and 1.
        for (auto& weight : layerWeights) {
            EXPECT_THAT(weight, testing::DoubleNear(0.0, 1.0));
        }
    }
}

TEST(FCNN, Forward) {
    hyper::FCNN model({5, 10, 10, 1});
    xt::xarray<double> input = {{1, 2, 3}, {4, 5, 6}};
    EXPECT_EQ(model.forward(input), 21);
}