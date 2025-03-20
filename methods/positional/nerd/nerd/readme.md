The repository contains refernce implementation of NERD proposed in 


**Node Representation Learning for Directed Graphs.**
M. Khosla, J. Leonhardt, W. Nejdl, A. Anand.
In ECML-PKDD 2019.


================================================================================

>***Basic Usage***


**Required**: gsl library for random number generation

**To Compile and Link** : g++ -lgsl -lgslcblas -lm -pthread *.cpp -o NERD


>  **For Large Graphs** : Increase the size of hash table in NERD.h (look for **hash_table_size**) accordingly.

The algorithm expects a **directed weighted edgelist**. If your graph is unweighted, please add 1 as weight for all edges. For undirected graphs, please add 2 edges ( in both directions) corresponding to each edge.
For bipartite undirected graphs, one can treat edges directed from the left set to the right set and doubling of edges is not required.


**Example command for training**  : ./NERD -train edgelist.txt -output1 hub.txt -output2 auth.txt -binary 0 -size 128-walkSize 5 -negative 5 -samples 10 -rho 0.025 -threads 20 -joint 1 -inputvertex 0

Parameters for training:

-train "file" : Use network data from "file" to train the model

-output1 "file": Use "file" to save the learnt source node (hub) embeddings

-output2 "file": Use "file" to save the learnt target node (authority) embeddings

-binary  :Save the learnt embeddings in binary moded; default is 0 (off)\n");
	    
-Size   : Set dimension of vertex embeddings; default is 100

-walkSize: Number of nodes of each type to be sampled (default is 3)

-negative : Number of negative examples

-samples : Set the number of training samples as <int>Million; default is 1
	    
-threads : Use <int> threads (default 20)
	
-rho : Set the starting learning rate; default is 0.025

-joint : if set 0 sample only hub-authority (in source walk) or authority-hub(in target walk) pairs , if 1 also sample hub-hub (in source walk) and 
         authority-authority pairs (default is 0)
         
-inputvertex : if set 0 Use the first vertex in the walk sample as the input word otherwise use the middle vertex in the sample walk (default is 0)

>**Evaluation**: The embeddings are evaluated for three tasks : Node Classification, Link Prediction and Graph Reconstruction. Respective evaluation scripts are evaluation/ multilabel_class_cv.py,
 evaluation/graph_reconstruction_sigmoid.py ,  evaluation/link_pred.py
 
> **Data**: Links to data and train-test psplits can be found under data/


