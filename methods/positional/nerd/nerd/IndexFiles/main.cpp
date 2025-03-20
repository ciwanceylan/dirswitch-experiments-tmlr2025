/*
 * main.cpp
 *
 *  Created on: Apr 17, 2018
 *      Author: khosla
 */


#include "index.h"

char network_file[MAX_STRING],edge_list[MAX_STRING],index_file[MAX_STRING];


int main(int argc, char **argv)
{
	int i;


	if ((i = ArgPos((char *)"-input", argc, argv)) > 0) strcpy(network_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-output1", argc, argv)) > 0) strcpy(index_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-output2", argc, argv)) > 0) strcpy(edge_list, argv[i + 1]);

	vertex = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex));
	InitHashTable();
	ReadData();

	return 0;
}
