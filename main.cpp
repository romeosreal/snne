#include <iostream>
#include <onn.h>

using namespace std;
using namespace onn;

int main()
{
  Topology topology;
  topology.addLayer(2); // Inputs
  topology.addLayer(3); // Hidden layer
  topology.addLayer(3); // Hidden layer
  topology.addLayer(1); // Output layer

  topology.print();

  Net net;
  net.setTopology(topology);
  net.setLearningRate(0.1);

  float input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  float output[4][1] = {{1}, {0}, {0}, {1}};

  for (int i = 0; i < 10000; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      net.forward(input[j]);
      net.backprop(output[j]);

      net.updateWeights();
    }
  }
  return 0;
}
