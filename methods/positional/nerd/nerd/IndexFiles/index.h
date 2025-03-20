/*
 * index.h
 *
 *  Created on: Apr 17, 2018
 *      Author: khosla
 */

#ifndef INDEX_H_
#define INDEX_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STRING 1000

extern int *vertex_hash_table;

const int hash_table_size = 30000000;
extern int max_num_vertices, num_vertices;

extern char network_file[MAX_STRING],edge_list[MAX_STRING],index_file[MAX_STRING];

extern int max_num_vertices, num_vertices;

struct ClassVertex {

	char *name;
};

 extern struct ClassVertex *vertex;



 int ArgPos(char *str, int argc, char **argv);
 /* Build a hash table, mapping each vertex name to a unique vertex id */
 unsigned int Hash(char *key);
 void InitHashTable();
 void InsertHashTable(char *key, int value);
 int SearchHashTable(char *key);

 /* Add a vertex to the vertex set */
 int AddVertex(char *name);


 /* Read network from the training file */
 void ReadData();


#endif /* INDEX_H_ */
