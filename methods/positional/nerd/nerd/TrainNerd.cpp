#include "NERD.h"

int walkSize = 3, in_vertex =0,joint=0;
//real error =0.0;
char init_src[MAX_STRING];char init_dst[MAX_STRING];



/* Initialize the vertex embedding and the context embedding */
void InitVector() {
    
    long long a, b;
    
    a = posix_memalign((void **) &emb_vertex, 128, (long long) num_vertices * dim * sizeof (real));
    if (emb_vertex == NULL) {
        printf("Error: memory allocation failed\n");
        exit(1);
    }
    for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
          emb_vertex[a * dim + b] = (rand() / (real) RAND_MAX - 0.5) / dim;
	 //emb_vertex[a*dim+b]=0;

    a = posix_memalign((void **) &emb_context, 128, (long long) num_vertices * dim * sizeof (real));
    if (emb_context == NULL) {
        printf("Error: memory allocation failed\n");
        exit(1);
    }
    
    for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
           // emb_context[a * dim + b] = (rand() / (real) RAND_MAX - 0.5) / dim;
	   emb_context[a*dim+b]=0;
// OverwriteVector();
}

// For NOn Random Initialization of Embeddings 
void OverwriteVector()
{
    int num_vertices, dim, a, b;
	char nameS[MAX_STRING], nameD[MAX_STRING], ch;
	real *vecS, *vecD;
	int vidS, vidD;
        
        
	FILE *fis, *fid;

	fis = fopen(init_src, "rb");
        fid = fopen(init_dst, "rb");
        
        if(fis == NULL || fid== NULL) 
        {
            printf("No Prior Embedding Files available\n");
            return;
            
        }
	

	fscanf(fis, "%d %d", &num_vertices, &dim);
        fscanf(fid, "%d %d", &num_vertices, &dim);
	vecS = (real *)malloc(dim * sizeof(real));
	vecD = (real *)malloc(dim * sizeof(real));
        
	for (a = 0; a < num_vertices; a++)
	{
		fscanf(fis, "%s%c", nameS, &ch);
                fscanf(fid, "%s%c", nameD, &ch);
                vidS = SearchHashTable(nameS);
                 vidD = SearchHashTable(nameD);
                 if(vidS != vidD)
                 {
                     printf("Error: Source and Embedding files do not match\n");
                         exit(1);
                 }
           
for (b = 0; b < dim; b++)
                    {
                        fread(&vecS[b], sizeof(real), 1, fis);
                        fread(&vecD[b], sizeof(real), 1, fid);
}     
                 if(vidS!=-1)
                {
                    for (b = 0; b < dim; b++)  
                    {
                      
                        emb_vertex[vidS * dim + b] = vecS[b] ;
                        emb_context[vidD * dim + b] = vecD[b] ;
                    }
                }	
	}
	free(vecS);
        free(vecD);
	fclose(fis);
	fclose(fid);
}

int SampleWeightedEdge(vector <double> weights)
{
	double total = 0.0;
	int size= weights.size();
	for(int i=0; i< size;i++)
	total += weights[i];

	double prob = randomLoc(total);
//printf("\n total : %lf", total);
//printf("\n prob : %lf", prob);
	
if(prob<=weights[0]) return 0;

	double weight1 = weights[0];
	double weight2 = weight1;

	for (int i=1; i< size; i++)
	{
		weight2 += weights[i];
			if(weight1<= prob && prob < weight2 )
				{
			//	printf("returned weight %d",i-1);
				return i-1;
				}	
			else
				weight1=weight2;

			}
return size-1;
}

/*Sample from a hub-authority random walk*/
int SampleHubWalk( int hid, int*hub, int* auth) {

    
int neighborLoc =0;

    for (int i = 0; i < walkSize; i++) {

        hub[i]=hid;
        double outdegree = vertex[hid].outvertices.size();
        
        if(outdegree==0){
           
            return i-1;
        } else {
        neighborLoc = SampleWeightedEdge(vertex[hid].outdegree);
        auth[i]=vertex[hid].outvertices[neighborLoc];
        }
        
        double indegree = vertex[auth[i]].invertices.size();
        
        if(indegree ==0)
        {
            
            return i;
        }
        else{
            neighborLoc = SampleWeightedEdge(vertex[auth[i]].indegree);
        hid=vertex[auth[i]].invertices[neighborLoc];
        }
        

    }
return walkSize;

}

/*Sample from a hub-authority random walk*/
int SampleAuthWalk(int aid, int *hub,int *auth ) {

    
    int neighborLoc;
    
try{
    for (int i = 0; i < walkSize; i++) {

        auth[i]=aid;

        int indegree = vertex[aid].invertices.size();

        if (indegree == 0) 
            return (i-1);
        else {
          neighborLoc = SampleWeightedEdge(vertex[aid].indegree);
            hub[i]=vertex[aid].invertices[neighborLoc];
        }

        int outdegree = vertex[hub[i]].outvertices.size();

        if (outdegree == 0) {
            return i;
            
        } else {
            neighborLoc = SampleWeightedEdge(vertex[hub[i]].outdegree);

            aid=vertex[hub[i]].outvertices[neighborLoc];
        }
       
    }
}
catch(...) { printf("Error in Sample AuthWalk"); }
    return walkSize;
}

void TrainHubWalk(unsigned long long seed, int sourceid, int* hub,int * auth) {

    int nxtHub = -1, u=-1, v, label, lv1,lv2, target1,target2;

    real *vec_error = (real *) calloc(dim, sizeof (real));

 int num_nodes= SampleHubWalk(sourceid,hub,auth);
 int middleindex =0;

if(num_nodes>=1)
{
	if(in_vertex)   middleindex = num_nodes/2;


   u = hub[middleindex];

}

else return;


    int lu = u * dim;
    for (int c = 0; c != dim; c++) vec_error[c] = 0;

    for (int i = 0; i < num_nodes; i++) {

        v = auth[i];


            nxtHub = hub[i];


        // Sample hub and authority and apply SGNS
        for (int d = 0; d != num_negative + 1; d++) {
            if (d == 0) {
                target1 = v;
                target2= nxtHub;
                label = 1;
            } else {
                target1 = neg_table_in[Rand(seed)];
		target2=neg_table_out[Rand(seed)];
                label = 0;
            }

            lv1 = target1 * dim;
	    lv2 = target2* dim;

           Update(&emb_vertex[lu], &emb_context[lv1], vec_error, label);

            if (i!=middleindex && joint)
              Update(&emb_vertex[lu], &emb_vertex[lv2], vec_error, label);
        }
    }

    for (int c = 0; c != dim; c++) emb_vertex[c + lu] += vec_error[c];


    free(vec_error);
}

void TrainAuthWalk(unsigned long long seed, int sourceid, int *hub, int*auth) {
    int nxtAuth = -1, v, u,label, lv1,lv2, target1, target2;
    int num_nodes= 0;
     num_nodes=SampleAuthWalk(sourceid,hub,auth);
    
    real *vec_error = (real *) calloc(dim, sizeof (real));
    
    int middleindex =0;

   if(num_nodes>=1)
   {

     if(in_vertex)
    	 middleindex = num_nodes/2;

      u = auth[middleindex];

   }    else return;

    int lu = u * dim;

    for (int c = 0; c != dim; c++) vec_error[c] = 0;

    for (int i = 0; i < num_nodes; i++) {

        v = hub[i];


            nxtAuth = auth[i];

        
        // Sample hub and authority and apply SGNS
        for (int d = 0; d != num_negative + 1; d++) {
            if (d == 0) {
                target1 = v;
                target2 =nxtAuth;
		label = 1;
                
            } else {
                target1 = neg_table_out[Rand(seed)];
                target2= neg_table_in[Rand(seed)];
		label = 0;
            }

            lv1 = target1 * dim;
            lv2= target2* dim;
            
           Update(&emb_context[lu], &emb_vertex[lv1], vec_error, label);

            if (i!=middleindex && joint)
                Update(&emb_context[lu], &emb_context[lv2], vec_error, label);
        }
    }

    for (int c = 0; c != dim; c++) emb_context[c + lu] += vec_error[c];


    free(vec_error);
}

/* Update embeddings */
void Update(real *vec_u, real *vec_v, real *vec_error, int label) {
    real x = 0,g;

    for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
    g = (label - FastSigmoid(x)) * rho;
    for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
    for (int c = 0; c != dim; c++) {
    //    printf("\n updating contexts");
        vec_v[c] += g * vec_u[c];
    //   error=g;
        
    }
}

/*Train HINGE Thread */
void *TrainHINGEThread(void *id) {
    
  //printf("Training....");

    long long count = 0, last_count = 0, i=0;

    unsigned long long seed = (long long) id;

  //  real* errorList= new real[total_samples/10000];
  //  fill(errorList,errorList+ (total_samples/10000),0);

    
    while (1) {
        //judge for exit
        if (count > total_samples / num_threads + 2) break;

        if (count - last_count > 10000) {
            current_sample_count += count - last_count;
            last_count = count;

         //   errorList[i]= error;
	    i++;
            printf("%cRho: %f  Progress: %.3lf%% ", 13, rho, (real) current_sample_count / (real) (total_samples + 1) * 100);
            fflush(stdout);
            rho = init_rho * (1 - current_sample_count / (real) (total_samples + 1));
            if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;


        }

        int sourceid = -1;


        if (coinThrow() > 0.5) {

        	sourceid = outdegree_table[randomIntLoc(unigram_table_size)];

         int* hub = new int[walkSize];int* auth = new int[walkSize];
        
            TrainHubWalk(seed,sourceid, hub,auth);
        delete []hub; delete []auth;
            
        }
        else {
        	sourceid = indegree_table[randomIntLoc(unigram_table_size)];

        int* hub = new int[walkSize];int* auth = new int[walkSize];
            fill(hub, hub+ walkSize, -1);
    fill(auth, auth + walkSize, -1);
            
            TrainAuthWalk(seed,sourceid, hub,auth);
            delete []hub; delete []auth;
            }

        count++;



    }

  // WriteError(errorList);
  // delete []errorList;
    pthread_exit(NULL);
}

void TrainHINGE() {
    long a;
    pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof (pthread_t));


    printf("--------------------------------\n");
    printf("Number of nodes of each type to be sampled : %d\n", walkSize);
    printf("Samples: %lldM\n", total_samples / 1000000);
    printf("Negative: %d\n", num_negative);
    printf("Dimension: %d\n", dim);
    printf("Initial rho: %lf\n", init_rho);
    printf("Input Vertex : %d\n",in_vertex);
    printf("Joint : %d\n",joint);
    printf("--------------------------------\n");

    InitHashTable();
    ReadData();
    
    
InitVector();
    
  

   InitNegTable_indegree();
   InitNegTable_outdegree();
  InitTable_indegree();
   InitTable_outdegree();
   InitSigmoidTable();

    rand_setup();

   
    clock_t start = clock();
    printf("--------------------------------\n");
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainHINGEThread, (void *) a);
   
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf\n", (double) (finish - start) / CLOCKS_PER_SEC);

    WriteEmb();



}
