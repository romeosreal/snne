#include <iostream>
#include <onn.h>

using namespace std;
using namespace onn;

int main()
{
  srand(42);

  Topology topology;
  topology.addLayer(2); // Inputs
  topology.addLayer(3); // Hidden layer
  topology.addLayer(1); // Output layer

  topology.print();
  int completed = 0;
  for (int i = 0; i < 1000; i++)
  {
    Net net;
    net.setTopology(topology);
    net.setLearningRate(0.1);
    net.setBatchSize(4);

    float input[4][2] = {{1, 1}, {1, 0}, {0, 1}, {0, 0}};
    float output[4][1] = {{1}, {0}, {0}, {1}};

    int32_t iter = 0;
    float loss;
    do
    {
      loss = 0;
      for (int j = 0; j < 4; j++)
      {
        net.forward(input[j]);
        net.backprop(output[j]);
        net.updateBatch();
        loss += net.loss();
        iter += 1;
        net.updateWeights();
      }

      if (loss < 1e-5)
      {
        completed += 1;
        break;
      }
    } while (iter < 5e5);
    if (iter < 5e5) cout << "Good" << endl;
    else cout << "Bad" << endl;
  }
  cout << completed << "/" << 1000 << endl;
  return 0;
}
