#ifndef __PLOT_ONN_H__
#define __PLOT_ONN_H__

#include <SFML/Graphics.hpp>
using namespace sf;

class Plot
{
  public:
    Plot();
    void create();
    void configure(int size, int windowSize);

    void update();
    void draw();
    void setPoint(int x, int y, float value);

    int size;
    int windowSize;

    float** data;
    RenderWindow window;
    Image canvas;
    Texture tex;
    Sprite sprite;
};

#endif
