#include <iostream>
#include <onn.h>

using namespace std;
using namespace onn;

int main()
{
  Topology topology;
  topology.addLayer(2); // Input  layer
  topology.addLayer(3); // Hidden layer
  topology.addLayer(1); // Output layer

  topology.print();

  Net net;
  net.setTopology(topology);
  net.setLearningRate(0.01);
  net.setBatchSize(4);
  net.configurePlot(400/* Plot quality */, 400/* Window size */);


  // Generate XOR problem dataset
  float **input = new float*[4];
  float **output = new float*[4];

  int id = 0;
  for (int x1 = 0; x1 < 2; x1++)
  {
    for (int x2 = 0; x2 < 2; x2++)
    {
      input[id] = new float[2];
      input[id][0] = x1;
      input[id][1] = x2;

      output[id] = new float[1];
      output[id][0] = x1 ^ x2;
      id++;
    }
  }


  // Train
  do {

    net.fitBatch(input, output, 4/*Dataset size*/);

    cout << "EPOCH: " << net.epoch << "\t LOSS: " << net.loss << endl;

  } while (net.loss > 1e-4);

  // Show plot
  net.show();

  return 0;
}
