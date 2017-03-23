#include "interactions.h"
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
// Required for input/output of PDB data
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
//using namespace std;
#include <sstream>

// texture and pixel objects
GLuint pbo = 0; // OpenGL pixel buffer object
GLuint tex = 0; // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource;

void render() {
  uchar4 *d_out = 0;
  cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
  cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,
                                       cuda_pbo_resource);
  kernelLauncher(d_out, d_vol, W, H, volumeSize, method, zs, theta,
                 threshold, dist, id);
  cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
  char title[128];
  sprintf(title, "Protein Visualizer : dataType =%d, method = %d,"
          " dist = %.1f, theta = %.1f", id, method, dist,
          theta);
  glutSetWindowTitle(title);
}

void draw_texture() {
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,
    GL_UNSIGNED_BYTE, NULL);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
  glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
  glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
  glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
  glEnd();
  glDisable(GL_TEXTURE_2D);
}

void display() {
  render();
  draw_texture();
  glutSwapBuffers();
}

void initGLUT(int *argc, char **argv) {
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(W, H);
  glutCreateWindow("Protein Visualizer");
#ifndef __APPLE__
  glewInit();
#endif
}

void initPixelBuffer() {
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, W*H*sizeof(GLubyte)* 4, 0,
               GL_STREAM_DRAW);
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
                               cudaGraphicsMapFlagsWriteDiscard);
}

void exitfunc() {
  if (pbo) {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
  }
  cudaFree(d_vol);
}

int main(int argc, char** argv) {
  float3 coords[ATOMS];
  float charge[ATOMS]; 
  // PDB files i/o
  std::ifstream posfile("positions.txt");
  std::ifstream chargefile("charge.txt");
  float x, y, z;
  int atomCount = 0;
  // store atomic coordinates from file in coords float3
  while (posfile >> x >> y >> z){
    coords[atomCount].x = x;
    coords[atomCount].y = y;
    coords[atomCount].z = z;
    atomCount = atomCount+1;
  }
  float w;
  int chargeCount = 0;
  // store partial charges in charge float
  while (chargefile >> w){
    charge[chargeCount] = w;
    chargeCount = chargeCount+1;
  }
  // Calc box dims from position file
  for (int i=  0; i < ATOMS; i = i + 1) {
    // find boxMin coordinates
    //maximum=(number>maximum)?number:maximum;
    boxMin.x=(coords[i].x<boxMin.x)?coords[i].x:boxMin.x;
    boxMin.y=(coords[i].y<boxMin.y)?coords[i].y:boxMin.y;
    boxMin.z=(coords[i].z<boxMin.z)?coords[i].z:boxMin.z;
    // find boxMax coordinates
    //maximum=(number>maximum)?number:maximum;
    boxMax.x=(coords[i].x>boxMax.x)?coords[i].x:boxMax.x;
    boxMax.y=(coords[i].y>boxMax.y)?coords[i].y:boxMax.y;
    boxMax.z=(coords[i].z>boxMax.z)?coords[i].z:boxMax.z;
  }
  // add 3 angstrom buffer so full spheres are always captured by rays
  boxMin.x = boxMin.x-3.0;
  boxMin.y = boxMin.y-3.0;
  boxMin.z = boxMin.z-3.0;
  boxMax.x = boxMax.x+3.0;
  boxMax.y = boxMax.y+3.0;
  boxMax.z = boxMax.z+3.0;
  // voxel dimensions for volume calcs
  voxDim.x = (boxMax.x-boxMin.x)/(float)NX;
  voxDim.y = (boxMax.y-boxMin.y)/(float)NY;
  voxDim.z = (boxMax.z-boxMin.z)/(float)NZ;

  float3 *d_coords = 0;
  float *d_charge = 0;
  float2 *vol = (float2*)calloc(NX*NY*NZ, sizeof(float2));
  cudaMalloc(&d_coords, ATOMS*sizeof(float3));
  cudaMalloc(&d_charge, ATOMS*sizeof(float));
  cudaMalloc(&d_vol, NX*NY*NZ*sizeof(float2));
  cudaMemcpy(d_coords,coords,ATOMS*sizeof(float3),cudaMemcpyHostToDevice);
  cudaMemcpy(d_charge,charge,ATOMS*sizeof(float),cudaMemcpyHostToDevice);
  volumeKernelLauncher(d_vol, d_coords, volumeSize, voxDim, boxMin, ATOMS, d_charge);
  cudaMemcpy(vol, d_vol, NX*NY*NZ*sizeof(float2), cudaMemcpyDeviceToHost);
  // check if anything happened
  printf("%f\n", vol[100].x);
  printInstructions();
  initGLUT(&argc, argv);
  createMenu();
  gluOrtho2D(0, W, H, 0);
  glutKeyboardFunc(keyboard);
  glutSpecialFunc(handleSpecialKeypress);
  glutDisplayFunc(display);
  initPixelBuffer();
  glutMainLoop();
  atexit(exitfunc);
  return 0;
}