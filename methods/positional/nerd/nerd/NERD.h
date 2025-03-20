#ifndef NERD_H_
#define	NERD_H_


#include<iostream>
#include<math.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include "fileoper.h"



using namespace std;

const int hash_table_size = 30000000;
const int unigram_table_size = 1e8;

const int sigmoid_table_size = 1000;




extern real init_rho, rho, error; //Learning rate
extern int in_vertex, joint;

int ArgPos(char *str, int argc, char **argv);

/* Fastly compute sigmoid function */
void InitSigmoidTable();
real FastSigmoid(real x);
void InitNegTable_indegree();
void InitNegTable_outdegree();
/* Generate a random integer */
int Rand(unsigned long long &seed);






extern int *vertex_hash_table, *neg_table, *neg_table_in, *neg_table_out, *indegree_table, *outdegree_table;

extern char init_src[MAX_STRING], init_dst[MAX_STRING];

void rand_setup();


void Update(real *vec_u, real *vec_v, real *vec_error, int label);
void *TrainHINGEThread(void *id);





/* Initialize the vertex embedding and the context embedding */
void InitVector();
/* Overwrite initialization with existing vectors*/
void OverwriteVector();
/* Sample negative vertex samples according to vertex degrees */
void InitNegTable();

void InitTable_indegree();
void InitTable_outdegree();
void InitNegTable_indegree();
void InitNegTable_outdegree();


double randomLoc(double size);
int randomIntLoc(int size);
int SampleHubWalk(int hid,int *hub,int *auth);
int SampleAuthWalk(int aid,int *hub,int *auth);
void TrainHINGE();
void TrainHubWalk(unsigned long long seed,int * hub,int* auth);
void TrainAuthWalk(unsigned long long seed, int * hub,int* auth);

double coinThrow ();


#endif


