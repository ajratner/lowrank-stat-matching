package lowrankparser;

public class DepDecoder {

	public String decodeProjective(DepInstance inst, double[][] arcScores,
			double[][][][] ntScores, boolean addLoss) {
		
		int N = inst.length;
		ParseForest pf = new ParseForest(N);
		for (int i = 0; i < N; ++i) {
			//pf.addItem(i, i, 0, i, -1, 0.0, null, null);
			pf.addItem(i, i, 1, i, -1, 0.0, null, null);
		}
		
		int[][] staticTypes = null;
		if (inst.ntFvs != null) {
		    staticTypes = getTypes(ntScores, N, inst.ntFvs[0].length);
		}
	    int[] deps = inst.deps;
        int[] labs = inst.labIds;

		for (int l = 1; l < N; ++l)
			for (int s = 0; s + l < N; ++s) {
				
				int t = s + l;
				
				double arcST = arcScores[s][t];
				double arcTS = arcScores[t][s];
				int typeST = 0, typeTS = 0;
				if (DepPipe.learnLabel) {
					typeST = staticTypes[s][t];
					typeTS = staticTypes[t][s];
					arcST += ntScores[s][typeST][1][0] + ntScores[t][typeST][1][1];
					arcTS += ntScores[s][typeTS][0][1] + ntScores[t][typeTS][0][0];
				}
                if (addLoss) {
                    arcST += deps[t] == s ? 0.0 : 1.0;
                    arcTS += deps[s] == t ? 0.0 : 1.0;
                    if (DepPipe.learnLabel) {
                        arcST += labs[t] == typeST ? 0.0 : 1.0;
                        arcTS += labs[s] == typeTS ? 0.0 : 1.0;
                    }
                }
				
				for (int r = s; r < t; ++r) {
					ParseForestItem x = pf.getItem(s, r, 1);
					ParseForestItem y = pf.getItem(t,r+1, 1);
					if (x == null || y == null) continue;
					
					pf.addItem(s, t, 0, r, typeST,
						arcST + x.score + y.score, x, y);
					pf.addItem(t, s, 0, r, typeTS,
						arcTS + x.score + y.score, x, y);
				}
				
				for (int r = s; r <= t; ++r) {	
					
					if (r != s) {
						ParseForestItem x = pf.getItem(s, r, 0);
						ParseForestItem y = pf.getItem(r, t, 1);
						if (x == null || y == null) continue;
						
						pf.addItem(s, t, 1, r, -1,
							x.score + y.score, x, y);
					}
					
					if (r != t) {
						ParseForestItem x = pf.getItem(r, s, 1);
						ParseForestItem y = pf.getItem(t, r, 0);
						if (x == null || y == null) continue;
						
						pf.addItem(t, s, 1, r, -1,
							x.score + y.score, x, y);
					}
				}
			}
		
		return pf.getBestParse();
	}
	
	public int[][] getTypes(double[][][][] ntScores, int N, int T) {
		
		int[][] staticTypes = new int[N][N];
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j) if (i != j) {
				
				int k = -1;
				double maxv = Double.NEGATIVE_INFINITY;				
				int toRight = i < j ? 1 : 0;	
				
				for (int t = 0; t < T; ++t) {
					double va = ntScores[i][t][toRight][0] + ntScores[j][t][toRight][1];
					if (va > maxv) {
						k = t;
						maxv = va;
					}
				}				
				staticTypes[i][j] = k;
			}
		return staticTypes;
	}

}

class ParseForest {
	
	public ParseForestItem[][][][] chart;
	public int K = 1;
	public int N;
	
	public ParseForest(int N) {
		chart = new ParseForestItem[N][N][2][K];
		this.N = N;
	}
	
	public void addItem(int s, int t, int comp, int r, int type, 
			double value, ParseForestItem left, ParseForestItem right) {
		
		if (chart[s][t][comp][K-1] == null || value > chart[s][t][comp][K-1].score) {
			ParseForestItem item = new ParseForestItem(s, t, comp, r, type, value, left, right);
			
			int i = K-1;
			while (i > 0 && (chart[s][t][comp][i-1] == null || value > chart[s][t][comp][i-1].score)) {
				chart[s][t][comp][i] = chart[s][t][comp][i-1];
				--i;
			}
			chart[s][t][comp][i] = item;
		}
	}
	
	public ParseForestItem getItem(int s, int t, int comp) {
		return chart[s][t][comp][0];
	}
	
	public String getBestParse() {
		return getDepString(chart[0][N-1][1][0]);
	}
	
	private String getDepString(ParseForestItem item) {
		
		if (item == null || item.s == item.t) return "";
		
		if (item.comp == 1) {
			String deps = (getDepString(item.left) + " " +
						   getDepString(item.right)).trim();
			return deps;
		} else {
			String deps = (getDepString(item.left) + " " +
					  	   getDepString(item.right)).trim();
			if (item.s < item.t)
				return (deps + " " + item.s + "|" + item.t + ":" + item.type).trim();
			else
				return (item.s + "|" + item.t + ":" + item.type + " " + deps).trim();
		}
	}
}

class ParseForestItem {
	
	int s, t, comp, r, type;
	double score;
	ParseForestItem left, right;
	
	public ParseForestItem(int s, int t, int comp, int r, int type,
			double value, ParseForestItem left, ParseForestItem right) {
		this.s = s;
		this.t = t;
		this.comp = comp;
		this.r = r;
		this.type = type;
		this.score = value;
		this.left = left;
		this.right = right;		
	}
}
