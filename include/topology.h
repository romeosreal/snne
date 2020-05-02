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
    void addLayer(uint16_t size);
    void print();
    uint16_t getLayersCount();
    uint16_t getLayerSize(uint16_t index);

    vector<int16_t> sequence;
  };
}

#endif
