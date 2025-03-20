Compile : g++ *.cpp -o indexFile

Run Example : ./indexFile.out -input inputfile_name -output1 index_file -output2 edge_list

inputfile_name  : Original edge list(weighted) to index 

for example :  
 
              a    b   12 

              c    d   23
index_file : name of the output file for indexing vertices . 

The output index_file for the above example would look like
 
                1   a
                2   b
                3   c
                4   d
                
edge_list : Edge list (weighted) with nove names replaced by integer indexes.For example
            
            1   2   12

            3   4   23