#include <onn.h>

using namespace onn;

Net::Net()
{

}

#define floatRand() ((float)(rand()) / (float)(RAND_MAX))

void Net::setTopology(Topology topology)
{

  srand(time(0));

  LAYERS_COUNT = topology.getLayersCount();
  layerSize = new uint16_t[LAYERS_COUNT];

  for (int i = 0; i < LAYERS_COUNT; i++)
    layerSize[i] = topology.getLayerSize(i);

  nodes = new Node*[LAYERS_COUNT];
  for (int i = 0; i < LAYERS_COUNT; i++)
  {
    nodes[i] = new Node[layerSize[i]];
    if (i != 0)
    {
      // For every node in every layer without input layer
      for (int j = 0; j < layerSize[i]; j++)
      {
        // Create and setup array for weights
        nodes[i][j].inputWeights = new float[layerSize[i - 1]];
        for (int k = 0; k < layerSize[i - 1]; k++)
        {
          nodes[i][j].inputWeights[k] = floatRand();
        }
        // Setup bias
        nodes[i][j].bias = floatRand();
      }
    }
    else
    {
      // Setup biases for input layer
      for (int j = 0; j < layerSize[i]; j++)
      {
        nodes[i][j].bias = floatRand();
      }
    }
  }
}

float* Net::forward(float* input)
{
  // Set first layer equals to input, and apply activation function
  for (int i = 0; i < layerSize[0]; i++)
  {
    nodes[0][i].value = input[i] + nodes[0][i].bias;
    nodes[0][i].activate();
  }

  // For every layer
  for (uint16_t previousLayer = 0, currentLayer = 1; currentLayer < LAYERS_COUNT; currentLayer++, previousLayer++)
  {
    uint16_t previousSize = layerSize[previousLayer];
    uint16_t currentSize = layerSize[currentLayer];

    // For every node in layer
    for (uint16_t i = 0; i < currentSize; i++) {
      // Set bias
      nodes[currentLayer][i].value = nodes[currentLayer][i].bias;

      // Connect current node with every node in previous layer
      for (uint16_t j = 0; j < previousSize; j++)
      {
        nodes[currentLayer][i].value += nodes[previousLayer][j].value * nodes[currentLayer][i].inputWeights[j];
      }

      // Apply activation function
      nodes[currentLayer][i].activate();
    }

  }

  // Parse last layer to float array and return it
  float* output = new float[layerSize[LAYERS_COUNT - 1]];
  for (uint16_t i = 0; i < layerSize[LAYERS_COUNT - 1]; i++)
  {
    cout << nodes[LAYERS_COUNT - 1][i].value << "->";
    output[i] = nodes[LAYERS_COUNT - 1][i].value;
  }

  return output;
}

#define sq(x) (x * x)

void Net::backprop(float *output)
{
  for (uint16_t i = 0; i < layerSize[LAYERS_COUNT - 1]; i++)
  {
    nodes[LAYERS_COUNT - 1][i].error = output[i] - nodes[LAYERS_COUNT - 1][i].value;
    cout << "ERROR:" << sq(nodes[LAYERS_COUNT - 1][i].error) << endl;
  }

  for (uint16_t nextLayer = LAYERS_COUNT - 1, currentLayer = nextLayer - 1; nextLayer > 0; nextLayer--, currentLayer--)
  {
    for (uint16_t i = 0; i < layerSize[currentLayer]; i++)
    {
      float total_error = 0;
      for (uint16_t j = 0; j < layerSize[nextLayer]; j++)
      {
        total_error += nodes[nextLayer][j].error * nodes[nextLayer][j].inputWeights[i];
      }

      nodes[currentLayer][i].error += nodes[currentLayer][i].getDerivative() * total_error;
    }
  }
}

void Net::setLearningRate(float lr)
{
  learningRate = lr;
}

void Net::updateWeights()
{
  for (int i = 1; i < LAYERS_COUNT; i++)
  {
    for (int j = 0; j < layerSize[i]; j++)
    {
      for (int k = 0; k < layerSize[i - 1]; k++)
      {
        nodes[i][j].inputWeights[k] += learningRate * nodes[i][j].error * nodes[i - 1][k].value;
      }

      nodes[i][j].error = 0;
    }
  }
}

Topology::Topology()
{

}

void Topology::print()
{
  cout << "(" << *sequence.begin() << ")";
  for (auto i = sequence.begin() + 1; i != sequence.end(); i++)
  {
    cout << "->(" << *i << ")";
  }
  cout << endl;
}

void Topology::addLayer(uint16_t size)
{
  sequence.push_back(size);
}

uint16_t Topology::getLayersCount()
{
  return this->sequence.size();
}

uint16_t Topology::getLayerSize(uint16_t index)
{
  return this->sequence[index];
}




float relu(float x)
{
  return (x > 0 ? x : 0);
}

float relu_derivative(float x)
{
  return (x > 0 ? 1 : 0);
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
