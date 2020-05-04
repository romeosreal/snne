#include <onn.h>

using namespace onn;

Net::Net()
{
  srand(time(0));

  momentum = 0.1;
  learningRate = 0.1;

  batchSize = 0;
  currentBatch = 0;

  epoch = 0;
}

#define floatRand() (((float)(rand()) / (float)(RAND_MAX)) - 0.5)

void Net::setTopology(Topology topology)
{
  LAYERS_COUNT = topology.getLayersCount();
  layerSize = new int[LAYERS_COUNT];

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
  for (int previousLayer = 0, currentLayer = 1; currentLayer < LAYERS_COUNT; currentLayer++, previousLayer++)
  {
    int previousSize = layerSize[previousLayer];
    int currentSize = layerSize[currentLayer];

    // For every node in layer
    for (int i = 0; i < currentSize; i++) {
      // Set bias
      nodes[currentLayer][i].value = nodes[currentLayer][i].bias;

      // Connect current node with every node in previous layer
      for (int j = 0; j < previousSize; j++)
      {
        nodes[currentLayer][i].value += nodes[previousLayer][j].value * nodes[currentLayer][i].inputWeights[j];
      }

      // Apply activation function
      nodes[currentLayer][i].activate();
    }

  }

  // Parse last layer to float array and return it
  float* output = new float[layerSize[LAYERS_COUNT - 1]];
  for (int i = 0; i < layerSize[LAYERS_COUNT - 1]; i++)
  {
    output[i] = nodes[LAYERS_COUNT - 1][i].value;
  }

  return output;
}

#define sq(x) (x * x)

float Net::calculateLoss()
{
  loss = 0;
  for (int batch = 0; batch < batchSize; batch++)
  {
    for (int i = 0; i < layerSize[LAYERS_COUNT - 1]; i++)
    {
      loss += sq(batchError[batch][LAYERS_COUNT - 1][i]);
    }
  }

  for (int i = 0; i < layerSize[LAYERS_COUNT - 1]; i++)
  {
    loss += sq(nodes[LAYERS_COUNT - 1][i].error);
  }

  return loss;
}


#define sign(x) (x > 0 ? 1 : -1)
void Net::backprop(float *output)
{
  for (int i = 0; i < layerSize[LAYERS_COUNT - 1]; i++)
  {
    nodes[LAYERS_COUNT - 1][i].error = (output[i] - nodes[LAYERS_COUNT - 1][i].value);// * sign(output[i] - nodes[LAYERS_COUNT - 1][i].value);
  }

  for (int nextLayer = LAYERS_COUNT - 1, currentLayer = nextLayer - 1; nextLayer > 0; nextLayer--, currentLayer--)
  {
    for (int i = 0; i < layerSize[currentLayer]; i++)
    {
      float total_error = 0;
      for (int j = 0; j < layerSize[nextLayer]; j++)
      {
        total_error += nodes[nextLayer][j].error * nodes[nextLayer][j].inputWeights[i];
      }

      nodes[currentLayer][i].error = total_error * nodes[currentLayer][i].getDerivative();
    }
  }

  this->updateBatch();
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

void Net::setBatchSize(int sz)
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


  for (int layer = 1; layer < LAYERS_COUNT; layer++)
  {
    for (int i = 0; i < layerSize[layer]; i++)
    {
      // Init array for weights delta
      float *dWeights = new float[layerSize[layer - 1]];
      for (int j = 0; j < layerSize[layer - 1]; j++) dWeights[j] = 0;

      // Init bias variable
      float dBias = 0;

      // For every batch + 1(current network)
      for (int batch = 0; batch <= batchSize; batch++)
      {
        // If processing saved batches
        if (batch != batchSize)
        {
          // Adjust weights
          for (int j = 0; j < layerSize[layer - 1]; j++)
          {
            dWeights[j] += lr * batchError[batch][layer][i] * batchValue[batch][layer - 1][j];
          }
          // Adjust bias
          dBias += lr2 * batchError[batch][layer][i];
        }
        else // else processing current network
        {
          // Adjust weights
          for (int j = 0; j < layerSize[layer - 1]; j++)
          {
            dWeights[j] += lr * nodes[layer][i].error * nodes[layer - 1][j].value;
          }
          // Adjust bias
          dBias += lr2 * nodes[layer][i].error;
        }
      }
      // Set all weights and bias
      for (int j = 0; j < layerSize[layer - 1]; j++)
      {
        float weight = nodes[layer][i].inputWeights[j];
        nodes[layer][i].inputWeights[j] = (weight + dWeights[j]) * (1. - momentum) + weight * momentum;
      }

      float bias = nodes[layer][i].bias;
      nodes[layer][i].bias = (bias + dBias) * (1. - momentum) + bias * momentum;
    }
  }

  currentBatch = 0;
}


void Net::setMomentum(float momentum)
{
  this->momentum = momentum;
}


void Net::fitBatch(float **input, float **output, int size)
{
  int *ids = new int[batchSize];

  for (int i = 0; i < batchSize; i++)
  {
    bool exist;
    do {
      exist = false;

      ids[i] = rand() % size;

      for (int j = 0; j < i; j++)
      {
        if (ids[i] == ids[j])
        {
          exist = true;
          break;
        }
      }

    } while (exist);

    epoch += 1;
  }

  for (int i = 0; i < batchSize; i++)
  {
    int id = ids[i];

    forward(input[id]);
    backprop(output[id]);
  }
  calculateLoss();
  updateWeights();
}

void Net::configurePlot(int size, int windowSize)
{
  #if SFML_SUPPORT
  plot.configure(size, windowSize);
  #endif
}

void Net::show()
{
  #if SFML_SUPPORT
  plot.create();
  int size = plot.size;

  for (int i = 0; i < size; i++)
  {
    for (int j = 0; j < size; j++)
    {
      float input[] = {(float)(i) / size, (float)(j) / size};
      float output = forward(input)[0];
      plot.setPoint(i, j, output);
    }
  }
  plot.update();
  while (plot.window.isOpen())
  plot.draw();
  #else
  cout << "ERROR: SFML is blocked, please, go to onn.h and set SFML_SUPPORT to 1 if you want to use SFML functionality" << endl;
  #endif
}
