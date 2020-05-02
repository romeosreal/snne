#ifndef __ONN_H__
#define __ONN_H__

#include <iostream>
#include <vector>
#include <cstdlib>
#include <time.h>

using namespace std;

namespace onn
{
  class Node
  {
  public:
    Node();
    void activate();
    float getDerivative();

    float *inputWeights;
    float *outputWeights;

    float bias;
    float error;
    float value;

    float (*activation)(float);
    float (*activation_derivative)(float);
  };

  class Topology
  {
  public:
    Topology();
    void addLayer(uint16_t size);
    void print();
    uint16_t getLayersCount();
    uint16_t getLayerSize(uint16_t index);

    vector<int16_t> sequence;
  };

  class Net
  {
  public:
    Net();
    void setTopology(Topology topology);
    float* forward(float *input);
    void backprop(float *output);
    void updateWeights();

    void setLearningRate(float lr);

    Node **nodes;

    uint16_t LAYERS_COUNT;
    uint16_t* layerSize;

    float learningRate = 0.1;
  };
}
#endif
