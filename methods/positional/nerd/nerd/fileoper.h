/* 
 * File:   fileoper.h
 * Author: khosla
 *
 * Created on March 29, 2018, 11:11 AM
 */

#ifndef FILEOPER_H
#define	FILEOPER_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define MAX_STRING 100

#define NEG_SAMPLING_POWER 0.75
#define SIGMOID_BOUND 6


typedef float real;                    // Precision of float numbers
using namespace std;


extern char network_file[MAX_STRING],embedding_file_hub[MAX_STRING],embedding_file_auth[MAX_STRING],error_log[MAX_STRING];


extern real *emb_vertex, *emb_context;

extern int is_binary, num_threads, walkSize, dim , num_negative, weighted;
extern long long total_samples, current_sample_count, num_edges;

extern int max_num_vertices, num_vertices;

struct inEdge
{
	int vertexid;
	int weight;
};

struct ClassVertex {
	double degree;
	double indegree_weight;
	double outdegree_weight;
	char *name;

	vector <int> invertices;
	vector <double> indegree;
	vector <int> outvertices;
	vector <double> outdegree;

	void AddInVertices(int vid, double weight)
	{
		invertices.push_back(vid);
		indegree.push_back(weight);
		indegree_weight += weight;

	}

	void AddOutVertices(int vid,double weight )
	{
		outvertices.push_back(vid);
		outdegree.push_back(weight);
		outdegree_weight += weight;

	}
};

extern struct ClassVertex *vertex;


/* Build a hash table, mapping each vertex name to a unique vertex id */
unsigned int Hash(char *key);
void InitHashTable();
void InsertHashTable(char *key, int value);
int SearchHashTable(char *key);

/* Add a vertex to the vertex set */
int AddVertex(char *name);


/* Read network from the training file */
void ReadData();
/*Write embedding vectors to respective files*/
void WriteEmb();

void WriteError(real* errorList);

#endif	/* FILEOPER_H */

