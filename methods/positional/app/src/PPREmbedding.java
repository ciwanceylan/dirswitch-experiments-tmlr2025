import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Map.Entry;


public class PPREmbedding extends SimilarityRanker {
	double[][] w;
	double[][] con;
	
	double[][] w_last;
	double[][] con_last;
	/**
	 * parameters
	 */
	public static int layer_size = 64;
	public static double alpha = 0.0025f;
	public static double starting_alpha = 0.05f;
	public static double jump_factor = 0.15f;
	public static int MAX_EXP = 5;
	public static Random r = new Random(0);
	// use to calculate e^i quickly
	public static double[] expTable;
	public static long next_random;
	public static int magic = 100;
	// negative samples
	public static int neg = 5;

	// #sampled paths-one hop
	public static int step = 10;
	public static int SAMPLE = 200;

	static {
		expTable = new double[1000];
		for (int i = 0; i < 1000; i++) {
			expTable[i] = (double) Math.exp((i / (double) 1000 * 2 - 1)
					* MAX_EXP); // Precompute the exp() table
			expTable[i] = expTable[i] / (expTable[i] + 1); // Precompute f(x) =
															// x / (x + 1)
		}
	}

	void set_params(int emb_dim, double jump_factor, int step) {
	    this.layer_size = emb_dim;
	    this.jump_factor = jump_factor;
	    this.step = step;
	}

	public static void rand_init(double[][] w) {
		for (int i = 0; i < w.length; i++) {
			double[] tmp = w[i];
			for (int j = 0; j < tmp.length; j++) {
				tmp[j] = (r.nextDouble() - 0.5) / layer_size;
			}
		}
	}
	static double global = 0;
	int size = 0;
	public void generateSimilairyMap() throws IOException {
		size = (int) Math.max(maxid + 1, ids.size());
		System.out.println(size);
		w = new double[size][layer_size];
		con = new double[size][layer_size];
		w_last = new double[size][layer_size];
		con_last = new double[size][layer_size];
		rand_init(w);
		// rand_init(con);

		Map<Long, List<Long>> g = new HashMap<Long, List<Long>>();

		for (Entry<Long, Set<Long>> ent : graph.entrySet()) {
			g.put(ent.getKey(), new ArrayList<Long>(ent.getValue()));
		}
	
		int iter = 20;
		alpha = starting_alpha;
		for (int kk = 0; kk < iter; kk++) {
			global = 0;
			for (int root = 0; root < size; root++) {
				List<Long> adjs = g.get((long) root);
				if (adjs == null || adjs.size() == 0)
					continue;
				for (int i = 0; i < SAMPLE; i++) {
					// sampled: from a to b
					int s = step;
					long id = -1;
					List<Long> tmp_adj = adjs;
					while (s-- > 0) {
						double jump = r.nextDouble();
						if (jump < jump_factor) {
							break;
						} else {
							id = tmp_adj.get(r.nextInt(tmp_adj.size()));
							tmp_adj = g.get((long) id);
						}
					}
					if (id != -1) {
						double[] e = new double[layer_size];
						// update as :word a, context b
						updateVector(w[root], con[(int) id], 1, e);

						for (int j = 0; j < neg; j++) {
							int nid = r.nextInt(size);
							if (nid == root)
								continue;
							List<Long> adj = g.get((long) nid);
							if (adj == null || adj.isEmpty())
                                continue;
							updateVector(w[root], con[(int) nid], 0, e);
						}

						for (int k = 0; k < layer_size; k++)
							w[root][k] += e[k];
					}

				}
			}
			System.out.println("iter:"+kk+":likelihood:"+global);
		}
	}

	private void updateVector(double[] w, double[] c, int label, double[] e) {
		double neg_g = calculateGradient(label, w, c);
		for (int i = 0; i < w.length; i++) {
			double tmp_c = c[i];
			double tmp_w = w[i];
			e[i] += neg_g * tmp_c;
			c[i] += neg_g * tmp_w;
		}
	}

	private static double calculateGradient(int label, double[] w, double[] c) {
		double f = 0, g;
		for (int i = 0; i < layer_size; i++)
			f += w[i] * c[i];
		if (f > MAX_EXP)
			g = (label - 1) * alpha;
		else if (f < -MAX_EXP)
			g = (label - 0) * alpha;
		else{
			double sigmoid = expTable[(int) ((f + MAX_EXP) * (1000 / MAX_EXP / 2))];
			g = (label - sigmoid)
					* alpha;
			if(label==1){
				global +=  Math.log(sigmoid);
			}else
				global +=  Math.log(1-sigmoid);
		}
		return g;
	}
	
	public void generateTopk(String path) throws IOException {
// 		BufferedWriter fw_ww = new BufferedWriter(new FileWriter(path+"_ww.txt"));
// 		BufferedWriter fw_wc = new BufferedWriter(new FileWriter(path+"_wc.txt"));
        File file = new File(path + "_vec.txt");
        file.createNewFile();

		BufferedWriter fw_vec = new BufferedWriter(new FileWriter(file));
		for (int i = 0; i < w.length; i++) {
			fw_vec.write(i + ",");
			for (int j = 0; j < layer_size; j++) {
				fw_vec.write(w[i][j] + ",");
			}
			fw_vec.write("\r\n");
		}
		fw_vec.flush();
		fw_vec.close();


		file = new File(path + "_vec_con.txt");
        file.createNewFile();

		fw_vec = new BufferedWriter(new FileWriter(file));
		for (int i = 0; i < con.length; i++) {
			fw_vec.write(i + ",");
			for (int j = 0; j < layer_size; j++) {
				fw_vec.write(con[i][j] + ",");
			}
			fw_vec.write("\r\n");
		}
		fw_vec.flush();
		fw_vec.close();
	}

	public static void main(String[] args) throws NumberFormatException,
			IOException {
		PPREmbedding ranker = new PPREmbedding();
		int emb_dim = new Integer(args[2]).intValue();
		double jump_factor = new Double(args[3]).doubleValue();
		int num_steps = new Integer(args[4]).intValue();
        ranker.set_params(emb_dim, jump_factor, num_steps);

        System.out.println(ranker.layer_size);
        System.out.println(ranker.jump_factor);
        System.out.println(ranker.step);

		ranker.readFromFile(args[0]);
		ranker.generateSimilairyMap();
		// ranker.generateSimilairyMatrix();
		System.out.println("generating top k file");
		ranker.generateTopk(args[1]);
	}
}
