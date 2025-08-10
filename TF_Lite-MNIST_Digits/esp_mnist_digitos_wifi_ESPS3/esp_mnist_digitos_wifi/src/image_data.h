#ifndef IMAGE_DATA_H_
#define IMAGE_DATA_H_

#include <Arduino.h> // Necessário para PROGMEM e uint8_t

// Declara a variável como externa.
// O compilador saberá que 'mnist_sample' existe em algum lugar.
// O tamanho não é necessário na declaração 'extern'.
extern const uint8_t mnist_sample[] PROGMEM;

#endif // IMAGE_DATA_H_