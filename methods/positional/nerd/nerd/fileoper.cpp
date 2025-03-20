#include "fileoper.h"



char network_file[MAX_STRING], embedding_file_hub[MAX_STRING],embedding_file_auth[MAX_STRING],error_log[MAX_STRING] ;

 
 
real *emb_vertex, *emb_context;
int weighted =1;

/* Read network from the training file */
void ReadData()
{
	FILE *fin;
	char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
	int vids,vidd;
	double weight;

	fin = fopen(network_file, "rb");
	if (fin == NULL)
	{
		printf("ERROR: network file not found!\n");
		exit(1);
	}
	num_edges = 0;
	while (fgets(str, sizeof(str), fin)) num_edges++;
	fclose(fin);
	printf("Number of edges: %lld          \n", num_edges);


	fin = fopen(network_file, "rb");
	num_vertices = 0;
	for (int k = 0; k < num_edges; k++)
	{
		fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

		if(!weighted)
			weight=1;

		if (k % 10000 == 0)
		{
			printf("Reading edges: %.3lf%%%c", k / (double)(num_edges + 1) * 100, 13);
			fflush(stdout);
		}



		vids = SearchHashTable(name_v1);
		if (vids == -1) vids = AddVertex(name_v1);
		vertex[vids].degree += weight;


		vidd = SearchHashTable(name_v2);
		if (vidd == -1) vidd = AddVertex(name_v2);
		vertex[vidd].degree += weight;



		vertex[vids].AddOutVertices(vidd, weight);
		vertex[vidd].AddInVertices(vids,weight);

 }
	fclose(fin);
	printf("Number of vertices: %d          \n", num_vertices);
        
}
/* Write Embeddings to file */
void WriteEmb()
{
	FILE *hub = fopen(embedding_file_hub, "wb");
	fprintf(hub, "%d %d\n", num_vertices, dim);
        
        FILE *auth = fopen(embedding_file_auth, "wb");
	fprintf(auth, "%d %d\n", num_vertices, dim);
        
	for (int a = 0; a < num_vertices; a++)
	{
		fprintf(hub, "%s ", vertex[a].name);
		if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(real), 1, hub);
		else for (int b = 0; b < dim; b++) fprintf(hub, "%lf ", emb_vertex[a * dim + b]);
		fprintf(hub, "\n");
                
	}
        for (int a = 0; a < num_vertices; a++)
	{
		fprintf(auth, "%s ", vertex[a].name);
		if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_context[a * dim + b], sizeof(real), 1, auth);
		else for (int b = 0; b < dim; b++) fprintf(auth, "%lf ", emb_context[a * dim + b]);
		fprintf(auth, "\n");
                
	}
        
	fclose(hub);
        fclose(auth);
}

/*Write Error after every 10,000 samples to the error_log file*/

void WriteError(real* errorList)
{
	

	FILE *error = fopen(error_log, "wb");
	if(error == NULL) return;

	fprintf(error, "%d %d\n", num_vertices, dim);
	for(int i=0; i< (total_samples/10000);i++)
	fprintf(error,"%lf\n",errorList[i]);

	fclose(error);
}

