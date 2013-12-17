package unit_test;

import utils.SVD;

public final class svdTester {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		double[] At = new double[] { 2, 3, 1, 4 };
		double[] Bt = new double[] { 1, 1, 3, 2, 10, 6 };
		
		double[] S = new double[2];
		double[] Ut = new double[2*2];
		double[] Vt = new double[2*3];
		
		int rank = SVD.lowRankSvd(At, Bt, 2, 3, 2, S, Ut, Vt);
		System.out.println(rank);
		for (int i = 0; i < rank; ++i)
			System.out.printf("%f ", S[i]);
		System.out.println();
		for (int i = 0; i < rank; ++i) {
			for (int j = 0; j < 2; ++j)
				System.out.printf("%f ", Ut[i*2+j]);
			System.out.println();
		}
		for (int i = 0; i < rank; ++i) {
			for (int j = 0; j < 3; ++j)
				System.out.printf("%f ", Vt[i*3+j]);
			System.out.println();
		}
		
	}

}
