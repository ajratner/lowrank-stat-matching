package utils;

import java.io.Serializable;

public class Rank1Component implements Serializable  {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public double sigma;
	public double[] u, v;
	
	public Rank1Component() {
		
	}
	
	public Rank1Component(double sigma, double[] u, double[] v) {
		this.sigma = sigma;
		this.u = u;
		this.v = v;
	}
}