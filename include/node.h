#ifndef __NODE_ONN_H__
#define __NODE_ONN_H__

namespace onn
{
  class Node
  {
  public:
    Node();
    void activate();
    float getDerivative();

    float *inputWeights;

    float bias;
    float error;
    float value;

    float (*activation)(float);
    float (*activation_derivative)(float);
  };
}

#endif
