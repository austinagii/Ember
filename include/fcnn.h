#ifndef FCNN_H
#define FCNN_H

#include <memory>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

namespace hyper {
  
  /**
   * @brief A fully connected neural network.
   */
  class FCNN {
    public:
      /**
       * @brief Construct a new FCNN
       * 
       * @param layerArchitecture The number of neurons per layer.
       */
      FCNN(const xt::xarray<std::size_t>& layerArchitecture) {
        const auto networkDepth = layerArchitecture.size();
        layerWeights = std::make_unique<std::vector<xt::xarray<double>>>();
        layerWeights->reserve(networkDepth - 1);

        // For each pair of adjacent layers, create a weight matrix where:
        // - Shape is (outputNeurons, inputNeurons) for each weight matrix
        // - Each row represents a neuron in the current layer
        // - Each column represents a connection to a neuron in the previous layer
        // - Weights are initialized with random values between 0 and 1
        for (size_t layerIdx = 1; layerIdx < networkDepth; ++layerIdx) {
            const auto outputNeurons = layerArchitecture[layerIdx];
            const auto inputNeurons = layerArchitecture[layerIdx-1];
            std::vector<std::size_t> weightDimensions = {outputNeurons, inputNeurons};
            layerWeights->push_back(xt::random::rand<double>(weightDimensions, 0, 1));
        }
      }

      ~FCNN() = default;

      double forward(const xt::xarray<double>& input) {
        return xt::sum(input)();
      }

      std::vector<xt::xarray<double>> get_weights() {
        return *layerWeights;
      }

    private:
      std::unique_ptr<std::vector<xt::xarray<double>>> layerWeights;
  };
}

// auto network = hyper::FCNN([5, 10, 10, 1]);

#endif // FCNN_H