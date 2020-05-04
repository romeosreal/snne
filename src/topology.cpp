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

void Topology::addLayer(int size)
{
  sequence.push_back(size);
}

int Topology::getLayersCount()
{
  return this->sequence.size();
}

int Topology::getLayerSize(int index)
{
  return this->sequence[index];
}
