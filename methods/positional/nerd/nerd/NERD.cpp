#include "NERD.h"




int main(int argc, char **argv) 
{
	int i;
	if (argc == 1) {
		printf("HINE: HITS Inspired Node Embedding\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse network data from <file> to train the model\n");
		printf("\t-output1 <file>\n");
		printf("\t\tUse <file> to save the learnt source node (hub) embeddings\n");
		printf("\t-output2 <file>\n");
		printf("\t\tUse <file> to save the learnt target node (authority) embeddings\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the learnt embeddings in binary moded; default is 0 (off)\n");
		printf("\t-size <int>\n");
		printf("\t\tSet dimension of vertex embeddings; default is 100\n");
		printf("\t-walkSize <int>\n");
		printf("\t\tNumber of nodes to be sampled of each type; default is 1\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 5\n");
		printf("\t-samples <int>\n");
		printf("\t\tSet the number of training samples as <int>Million; default is 1\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");
		printf("\t-rho <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
		printf("\t-weighted <int>\n");
		printf("\t\tUse input as unweighted graph if set as 0 n");
		printf("\nExamples:\n");
		printf("./HINE -train net.txt -output1 hub.txt -output2 auth.txt -binary 0 -size 50 -walkSize 2 -negative 5 -samples 100 -rho 0.025 -threads 20\n\n");
		return 0;
	}
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(network_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-output1", argc, argv)) > 0) strcpy(embedding_file_hub, argv[i + 1]);
    if ((i = ArgPos((char *)"-output2", argc, argv)) > 0) strcpy(embedding_file_auth, argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-walkSize", argc, argv)) > 0) walkSize = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-errorLog", argc, argv)) > 0) strcpy(error_log, argv[i + 1]);
	if ((i = ArgPos((char *)"-inputvertex", argc, argv)) > 0) in_vertex =  atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-joint", argc, argv)) > 0) joint =  atoi(argv[i + 1]);
// Arguments used for Non Random Initialization of Embeddings 
//if ((i = ArgPos((char *)"-initSrc", argc, argv)) > 0) strcpy(init_src, argv[i + 1]);
 //   if ((i = ArgPos((char *)"-initDst", argc, argv)) > 0) strcpy(init_dst, argv[i + 1]);
    if ((i = ArgPos((char *)"-weighted", argc, argv)) > 0) weighted = atoi(argv[i+1]);

	total_samples *= 1000000;
	rho = init_rho;
	vertex = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex));
	TrainHINGE();
//        free(vertex);
	return 0;
}
