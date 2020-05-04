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
    void fitBatch(float **input, float **output, int size);

    void configurePlot(int size, int windowSize);
    void show();

    void setBatchSize(int sz);
    void setLearningRate(float lr);
    void setMomentum(float momentum);

    float calculateLoss();
    float loss;

    uint32_t epoch;

    Node **nodes;

    int LAYERS_COUNT;
    int* layerSize;

    float learningRate;
    float momentum;

    int batchSize;
    int currentBatch;

    float ***batchValue;
    float ***batchError;

    #if SFML_SUPPORT
    Plot plot;
    #endif
  };
}

#endif
