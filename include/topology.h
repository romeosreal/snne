#ifndef __TOPOLOGY_ONN_H__
#define __TOPOLOGY_ONN_H__

#include <iostream>
#include <vector>

using namespace std;

namespace onn
{
  class Topology
  {
  public:
    Topology();
    void addLayer(int size);
    void print();
    int getLayersCount();
    int getLayerSize(int index);

    vector<int16_t> sequence;
  };
}

#endif
