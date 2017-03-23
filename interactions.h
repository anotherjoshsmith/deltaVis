#ifndef INTERACTIONS_H
#define INTERACTIONS_H
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif
#include <vector_types.h>
#define W 1000
#define H 1000
#define DELTA 5 // pixel increment for arrow keys
#define NX 256
#define NY 256
#define NZ 256

int id = 1; // 1 = atom position, 2 = partial charge
int method = 2; // 1 = slice, 2 = raycast
const int3 volumeSize = { NX, NY, NZ }; // size of volumetric data grid
float2 *d_vol; // pointer to device array for storing volume data
const int ATOMS = 570; // number of protein atoms
float3 *d_coords;
float *d_charge;
float zs = NZ; // distance from origin to source
float dist = 0.f, theta = 0.f, threshold = 0.7f;
float3 boxMax = { 0.0f, 0.0f, 0.0f };
float3 boxMin = { 1000.0f, 1000.0f, 1000.0f };
float3 voxDim;


void mymenu(int value) {
  switch (value) {
  case 0: return;
  case 1: id = 1; break; // position
  case 2: id = 2; break; // partial charge
  }
  glutPostRedisplay();
}

void createMenu() {
  glutCreateMenu(mymenu); // Object selection menu
  glutAddMenuEntry("Slice Technique", 0); // menu title
  glutAddMenuEntry("Atomic Positions", 1); // id = 1 -> positions
  glutAddMenuEntry("Partial Charge", 2); // id = 2 -> partial charge
  glutAttachMenu(GLUT_RIGHT_BUTTON); // right-click for menu
}

void keyboard(unsigned char key, int x, int y) {
  if (key == '+') zs -= DELTA; // move source closer (zoom in)
  if (key == '-') zs += DELTA; // move source farther (zoom out)
  if (key == 'd') --dist; // decrease slice distance
  if (key == 'D') ++dist; // increase slice distance
  if (key == 'z') zs = NZ, theta = 0.f, dist = 0.f; // reset values
  if (key == 's') method = 1; // slice
  if (key == 'r') method = 2; // raycast
  if (key == 27) exit(0);
  glutPostRedisplay();
}

void handleSpecialKeypress(int key, int x, int y) {
  if (key == GLUT_KEY_LEFT) theta -= 0.1f; // rotate left
  if (key == GLUT_KEY_RIGHT) theta += 0.1f; // rotate right
  if (key == GLUT_KEY_UP) threshold += 0.1f; // inc threshold (thick)
  if (key == GLUT_KEY_DOWN) threshold -= 0.1f; // dec threshold (thin)
  glutPostRedisplay();
}

void printInstructions() {
  printf("Protein Visualizer\n"
         "Controls:\n"
         "Slice render mode                           : s\n"
         "Raycast mode                                : r\n"
         "Zoom out/in                                 : -/+\n"
         "Rotate view                                 : left/right\n"
         "Decr./Incr. Offset (intensity in slice mode): down/up\n"
         "Decr./Incr. distance (only in slice mode)   : d/D\n"
         "Reset parameters                            : z\n"
         "Right-click for object selection menu\n");
}

#endif