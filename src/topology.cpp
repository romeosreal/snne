#include <onn.h>

using namespace onn;

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
