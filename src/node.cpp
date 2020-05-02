#include <node.h>

using namespace onn;

float relu(float x)
{
  return (x > 0 ? x : 0.01 * x);
}

float relu_derivative(float x)
{
  return (x > 0 ? 1 : 0.01);
}

Node::Node()
{
  activation = relu;
  activation_derivative = relu_derivative;

  error = 0;
}

void Node::activate()
{
  value = activation(value);
}

float Node::getDerivative()
{
  return activation_derivative(value);
}
