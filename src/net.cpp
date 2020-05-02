#include <onn.h>

using namespace onn;

Net::Net()
{

}

#define floatRand() (((float)(rand()) / (float)(RAND_MAX)) - 0.5)

void Net::setTopology(Topology topology)
{
  LAYERS_COUNT = topology.getLayersCount();
  layerSize = new uint16_t[LAYERS_COUNT];

  for (int i = 0; i < LAYERS_COUNT; i++)
    layerSize[i] = topology.getLayerSize(i);

  nodes = new Node*[LAYERS_COUNT];
  for (int i = 0; i < LAYERS_COUNT; i++)
  {
    nodes[i] = new Node[layerSize[i]];
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
}

float* Net::forward(float* input)
{
  // Set first layer equals to input
  for (int i = 0; i < layerSize[0]; i++)
  {
    nodes[0][i].value = input[i];
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
    output[i] = nodes[LAYERS_COUNT - 1][i].value;
  }

  return output;
}

#define sq(x) (x * x)

float Net::loss()
{
  float loss = 0;
  for (uint16_t batch = 0; batch < batchSize; batch++)
  {
    for (uint16_t i = 0; i < layerSize[LAYERS_COUNT - 1]; i++)
    {
      loss += sq(batchError[batch][LAYERS_COUNT - 1][i]);
    }
  }

  for (uint16_t i = 0; i < layerSize[LAYERS_COUNT - 1]; i++)
  {
    loss += sq(nodes[LAYERS_COUNT - 1][i].error);
  }

  return loss;
}

#define beta 0.7

void Net::backprop(float *output)
{
  for (uint16_t i = 0; i < layerSize[LAYERS_COUNT - 1]; i++)
  {
    nodes[LAYERS_COUNT - 1][i].error = (output[i] - nodes[LAYERS_COUNT - 1][i].value);
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

      nodes[currentLayer][i].error = total_error * nodes[currentLayer][i].getDerivative() * beta + (1 - beta) * nodes[currentLayer][i].error;
    }
  }
}

void Net::updateBatch()
{
  if (currentBatch >= batchSize) return;

  for (int layer = 0; layer < LAYERS_COUNT; layer++)
  {
    for (int i = 0; i < layerSize[layer]; i++)
    {
      batchValue[currentBatch][layer][i] = nodes[layer][i].value;
      batchError[currentBatch][layer][i] = nodes[layer][i].error;
    }
  }

  currentBatch += 1;
}

void Net::setLearningRate(float lr)
{
  learningRate = lr;
}

void Net::setBatchSize(uint16_t sz)
{
  if (sz < 1)
  {
    cout << "! INCORRECT BATCH SIZE !" << endl;
    return;
  }

  if (sz == 1) return;

  batchSize = sz - 1;

  batchValue = new float**[batchSize];
  batchError = new float**[batchSize];
  for (int batch = 0; batch < batchSize; batch++)
  {
    batchValue[batch] = new float*[LAYERS_COUNT];
    batchError[batch] = new float*[LAYERS_COUNT];
    for (int layer = 0; layer < LAYERS_COUNT; layer++)
    {
      batchValue[batch][layer] = new float[layerSize[layer]];
      batchError[batch][layer] = new float[layerSize[layer]];
    }
  }
}

void Net::updateWeights()
{

  float lr = learningRate;
  float lr2 = lr;

  for (int batch = 0; batch < batchSize; batch++)
  {
    for (int layer = 1; layer < LAYERS_COUNT; layer++)
    {
      for (int i = 0; i < layerSize[layer]; i++)
      {
        for (int j = 0; j < layerSize[layer - 1]; j++)
        {
          nodes[layer][i].inputWeights[j] += lr * batchError[batch][layer][i] * batchValue[batch][layer - 1][j];
        }
        nodes[layer][i].bias += lr2 * batchError[batch][layer][i];
      }
    }
  }

  for (int layer = 1; layer < LAYERS_COUNT; layer++)
  {
    for (int i = 0; i < layerSize[layer]; i++)
    {
      for (int j = 0; j < layerSize[layer - 1]; j++)
      {
        nodes[layer][i].inputWeights[j] += lr * nodes[layer][i].error * nodes[layer - 1][j].value;
      }
      nodes[layer][i].bias += lr2 * nodes[layer][i].error;
    }
  }

  currentBatch = 0;
}
