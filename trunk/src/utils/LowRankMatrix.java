package utils;

import java.io.Serializable;
import java.util.ArrayList;

import lowrankparser.Alphabet;

public class LowRankMatrix implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public static int R = 100;
		
	Rank1Component[] list = new Rank1Component[R * 2 + 1];
	double scale;
	int size;

	public LowRankMatrix() {
		size = 0;
		scale = 1.0;
	}
	
	public int size() { return size; }
	public double scale() { return scale; }
	
	public void clear() { size = 0; scale = 1.0; }
	
	public void rescale(double coeff) {
		scale *= coeff;
	}
    
    public void resetScale() {
        for (int i = 0; i < size; ++i)
            list[i].sigma *= scale;
        scale = 1.0;
    }

	public double traceNorm() {
		double sum = 0.0;
		for (int i = 0; i < size; ++i)			
			sum += list[i].sigma;
		return sum * scale;
	}
    
	public void add(double sigma, double[] u, double[] v) {
		list[size] = new Rank1Component(sigma / scale, u, v);
		size++;
		if (size == R * 2 + 1) {
			compress();
		}
	}
	
	private void compress() {
		int n = list[0].u.length;
		
		double[] At = new double[size*n], Bt = new double[size*n];
		for (int i = 0; i < size; ++i)
			for (int j = 0; j < n; ++j)
				At[i*n+j] = list[i].u[j] * list[i].sigma * scale;
		for (int i = 0; i < size; ++i)
			for (int j = 0; j < n; ++j)
				Bt[i*n+j] = list[i].v[j];
		
		double[] S = new double[size], Ut = new double[size*n], Vt = new double[size*n];
		int rank = SVD.lowRankSvd(At, Bt, n, n, size, S, Ut, Vt);
		
		size = rank > R ? R : rank;
		if (size > 0) {
            double sum = 0;
            for (int i = 0; i < rank; ++i) sum += S[i];
			System.out.printf("  Truncated A via SVD  (Rank: %d Sigma: max=%f cut=%f min=%f sum=%f )%n",
					rank, S[0], S[size], S[rank-1], sum);
			for (int i = 0; i < size; ++i) {
				double[] u = new double[n];
				double[] v = new double[n];
				for (int j = 0; j < n; ++j)
					u[j] = Ut[i*n+j];
				for (int j = 0; j < n; ++j)
					v[j] = Vt[i*n+j];
				list[i] = new Rank1Component(S[i], u, v);
			}
		}
		scale = 1.0;
	}
	
    public double min() {
        double m = Double.POSITIVE_INFINITY;
        for (int i = 0; i < size; ++i)
            if (m > list[i].sigma) m = list[i].sigma;
        return m * scale;
    }

    public double max() {
        double m = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < size; ++i)
            if (m < list[i].sigma) m = list[i].sigma;
        return m * scale;
    }
    
    public double getScore(SparseMatrix head, SparseMatrix child) {
    	double sum = 0;
        for (int i = 0; i < size; ++i) {
            Rank1Component c = list[i];
            sum += c.sigma * head.dotProduct(c.u) *
                    child.dotProduct(c.v);
        }
    	return sum * scale;
    }
    
    public double dotProduct(SparseMatrix m) {
    	double sum = 0;
    	for (int i = 0; i < size; ++i) {
    		Rank1Component c = list[i];
    		sum += c.sigma * m.leftRightMultiply(c.u, c.v);
    	}
    	return sum * scale;
    }
    
}


