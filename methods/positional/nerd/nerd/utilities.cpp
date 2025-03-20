#include "NERD.h"





int is_binary = 0, num_threads = 20, order = 2, dim = 100, num_negative = 5;

int max_num_vertices = 1000, num_vertices = 0;
long long total_samples = 1, current_sample_count = 0, num_edges = 0;
real init_rho = 0.025, rho;
 struct ClassVertex *vertex;
 
 int *vertex_hash_table, *neg_table, *neg_table_in, *neg_table_out, *indegree_table, *outdegree_table;




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





 /* Fastly generate a random integer */
 int Rand(unsigned long long &seed)
 {
 	seed = seed * 25214903917 + 11;
 	return (seed >> 16) % unigram_table_size;
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
	vertex[num_vertices].degree = 0;
	vertex[num_vertices].indegree_weight = 0;
	vertex[num_vertices].outdegree_weight = 0;
	num_vertices++;
	if (num_vertices + 2 >= max_num_vertices)
	{
		max_num_vertices += 1000;
		vertex = (struct ClassVertex *)realloc(vertex, max_num_vertices * sizeof(struct ClassVertex));
	}
	InsertHashTable(name, num_vertices - 1);
	return num_vertices - 1;
}



/* Sample negative vertex samples according to vertex degrees */
void InitNegTable()
{
	double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	neg_table = (int *)malloc(unigram_table_size * sizeof(int));
	for (int k = 0; k != num_vertices; k++) sum += pow(vertex[k].degree, NEG_SAMPLING_POWER);
	for (int k = 0; k != unigram_table_size; k++)
	{
		if ((double)(k + 1) / unigram_table_size > por)
		{
			cur_sum += pow(vertex[vid].degree, NEG_SAMPLING_POWER);
			por = cur_sum / sum;
			vid++;
		}
		neg_table[k] = vid - 1;
	}
}
/* Sample initial vertex with respect to indegree */
void InitTable_indegree()
{
	double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	indegree_table= (int *)malloc(unigram_table_size * sizeof(int));
	for (int k = 0; k != num_vertices; k++) sum += vertex[k].indegree_weight;
	for (int k = 0; k != unigram_table_size; k++)
	{
		if ((double)(k + 1) / unigram_table_size > por)
		{
			cur_sum += vertex[vid].indegree_weight;
			por = cur_sum / sum;
			vid++;
		}
		indegree_table[k] = vid - 1;
	}
}

/* Sample initial vertex with respect to outdegree */
void InitNegTable_indegree()
{
	double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	neg_table_in= (int *)malloc(unigram_table_size * sizeof(int));
	for (int k = 0; k != num_vertices; k++) sum += pow(vertex[k].indegree_weight, NEG_SAMPLING_POWER);
	for (int k = 0; k != unigram_table_size; k++)
	{
		if ((double)(k + 1) / unigram_table_size > por)
		{
			cur_sum += pow(vertex[vid].indegree_weight, NEG_SAMPLING_POWER);
			por = cur_sum / sum;
			vid++;
		}
		neg_table_in[k] = vid - 1;
	}
}

void InitTable_outdegree()
{
	double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	outdegree_table= (int *)malloc(unigram_table_size * sizeof(int));
	for (int k = 0; k != num_vertices; k++) sum += vertex[k].outdegree_weight;
	for (int k = 0; k !=unigram_table_size; k++)
	{
		if ((double)(k + 1) / unigram_table_size > por)
		{
			cur_sum += vertex[vid].outdegree_weight;
			por = cur_sum / sum;
			vid++;
		}
		outdegree_table[k] = vid - 1;
	}
}

/* Sample negative vertex samples according to vertex outdegrees */
void InitNegTable_outdegree()
{
	double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	neg_table_out = (int *)malloc(unigram_table_size * sizeof(int));
	for (int k = 0; k != num_vertices; k++) sum += pow(vertex[k].outdegree_weight, NEG_SAMPLING_POWER);
	for (int k = 0; k != unigram_table_size; k++)
	{
		if ((double)(k + 1) / unigram_table_size > por)
		{
			cur_sum += pow(vertex[vid].outdegree_weight, NEG_SAMPLING_POWER);
			por = cur_sum / sum;
			vid++;
		}
		neg_table_out[k] = vid - 1;
	}
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

