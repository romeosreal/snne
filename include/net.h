#ifndef __NET_ONN_H__
#define __NET_ONN_H__

#include <iostream>
#include <cstdlib>
#include <math.h>
#include <time.h>

using namespace std;

namespace onn
{
  class Net
  {
  public:
    Net();
    void setTopology(Topology topology);
    float* forward(float *input);
    void backprop(float *output);
    void updateWeights();
    void updateBatch();

    void setBatchSize(uint16_t sz);
    void setLearningRate(float lr);

    float loss();

    Node **nodes;

    uint16_t LAYERS_COUNT;
    uint16_t* layerSize;

    float learningRate = 0.1;
    uint16_t batchSize = 0;
    uint16_t currentBatch = 0;

    float ***batchValue;
    float ***batchError;
  };
}

#endif
