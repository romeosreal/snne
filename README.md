# SNNE
Small Neural Network Engine writed on C++

It can be used for MLP training on simple tasks

## Species
One thread.

Batch training.

As activation function uses LeakyReLU, but you can change it.

Can create small plot with Neural Network decision space, like this (XOR problem solving)
![Plot with XOR problem](https://github.com/romeosreal/snne/blob/master/docs/image01.png)

## Building

If you want to build it without SFML you need to use 
```make```

If you want to build it with SFML you need to use
```make sfml```
And set 
```#define SFML_SUPPORT 1```
In include/onn.h file

## Start

1. Make project 
2. Start app.o using, e.g ```./app.o```
