#include "index.h"



int max_num_vertices = 1000, num_vertices = 0;
long long current_sample_count = 0, num_edges = 0;

 struct ClassVertex *vertex;

 int *vertex_hash_table;

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

 void InitHashTable()
 {
 	vertex_hash_table = (int *)malloc(hash_table_size * sizeof(int));
 	for (int k = 0; k != hash_table_size; k++) vertex_hash_table[k] = -1;
 }

 void InsertHashTable(char *key, int value)
 {
 	int addr = Hash(key);
 	while (vertex_hash_table[addr] != -1) addr = (addr + 1) % hash_table_size;
 	vertex_hash_table[addr] = value;
 }


/* Build a hash table, mapping each vertex name to a unique vertex id */
 unsigned int Hash(char *key)
 {
 	unsigned int seed = 131;
 	unsigned int hash = 0;
 	while (*key)
 	{
 		hash = hash * seed + (*key++);
 	}
 	return hash % hash_table_size;
 }
 
 int SearchHashTable(char *key)
{
	int addr = Hash(key);
	while (1)
	{
		if (vertex_hash_table[addr] == -1) return -1;
		if (!strcmp(key, vertex[vertex_hash_table[addr]].name)) return vertex_hash_table[addr];
		addr = (addr + 1) % hash_table_size;
	}
	return -1;
}

 /* Add a vertex to the vertex set */
int AddVertex(char *name)
{
	int length = strlen(name) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vertex[num_vertices].name = (char *)calloc(length, sizeof(char));
	strncpy(vertex[num_vertices].name, name, length-1);
	
	num_vertices++;
	if (num_vertices + 2 >= max_num_vertices)
	{
		max_num_vertices += 1000;
		vertex = (struct ClassVertex *)realloc(vertex, max_num_vertices * sizeof(struct ClassVertex));
	}
	InsertHashTable(name, num_vertices-1);
	return num_vertices-1 ;
}


void ReadData()
{

    
    FILE *fin;
    FILE *fout1 = fopen(edge_list, "wb");
    FILE *fout2 = fopen(index_file, "wb");


	char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
	int vids,vidd;
	int weight;

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
	num_vertices = 1;
	for (int k = 0; k < num_edges; k++)
	{
		fscanf(fin, "%s %s %d", name_v1, name_v2, &weight);

		if (k % 10000 == 0)
		{
			printf("Reading edges: %.3lf%%%c", k / (double)(num_edges + 1) * 100, 13);
			fflush(stdout);
		}



		vids = SearchHashTable(name_v1);
		if (vids == -1)

			{
			vids = AddVertex(name_v1);
			fprintf(fout2, "%d %s\n", vids, name_v1);
			}


		

		vidd = SearchHashTable(name_v2);
		
		if (vidd == -1)
			{
			vidd = AddVertex(name_v2);
			fprintf(fout2, "%d %s\n", vidd, name_v2);
			}


		fprintf(fout1, "%d %d %d\n", vids, vidd, weight);

 }
	fclose(fin);
	fclose(fout1);
	fclose(fout2);
	printf("Number of vertices: %d          \n", num_vertices);
        
}
