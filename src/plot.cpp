#include <plot.h>

Plot::Plot()
{
  configure(40, 400);
}

void Plot::configure(int size, int windowSize)
{
  this->size = size;
  this->windowSize = windowSize;

  data = new float*[size];
  for (int i = 0; i < size; i++)
  {
    data[i] = new float[size];
    for (int j = 0; j < size; j++)
    {
      data[i][j] = 0;
    }
  }
}

void Plot::create()
{
  window.create(VideoMode(windowSize, windowSize), "Plot");
  canvas.create(windowSize, windowSize);
}

void Plot::setPoint(int x, int y, float value)
{
  data[x][y] = value;
}

#define sq(x) (x * x)

void Plot::update()
{

  int scale = windowSize / size;
  sprite.setPosition(-scale / 2, -scale / 2);

  for (int i = scale; i < windowSize; i++)
  {
    for (int j = scale; j < windowSize; j++)
    {
      float dst1 = sq((j % scale) * (i % scale));
      float dst2 = sq((scale - j % scale) * (i % scale));
      float dst3 = sq((scale - i % scale) * (j % scale));
      float dst4 = sq((scale - j % scale) * (scale - i % scale));


      float value = 255 * (float)( \
      data[(i / scale) + 0][(j / scale) + 0] * dst1 + \
      data[(i / scale) + 0][(j / scale) - 1] * dst2 + \
      data[(i / scale) - 1][(j / scale) + 0] * dst3 + \
      data[(i / scale) - 1][(j / scale) - 1] * dst4 ) / \
      (float)(dst1 + dst2 + dst3 + dst4);

      if (value < 0) value = 0;
      if (value > 255) value = 255;
      canvas.setPixel(i, j, Color(value, value, value));
    }
  }
  tex.loadFromImage(canvas);
  sprite.setTexture(tex);
}

void Plot::draw()
{
  if (window.isOpen())
  {
    Event event;
    while (window.pollEvent(event))
    {
        if (event.type == Event::Closed)
            window.close();
    }

    window.clear();
    window.draw(sprite);
    window.display();
  }
}
