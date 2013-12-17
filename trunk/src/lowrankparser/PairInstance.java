package lowrankparser;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.io.*;

import utils.SparseMatrix;


// A: new instance type, simpler...
public class PairInstance implements Serializable {

	private static final long serialVersionUID = 1L;

  public int length;
  public SparseMatrix[] vecs;
  public boolean label;
	
  public PairInstance() {}
  //public PairInstance(int length) { this.length = length; }
  public PairInstance(float[] v1, float[] v2, boolean lbl) {
    this.length = v1.length;
    this.label = lbl;
    SparseMatrix[] v = new SparseMatrix[2];
    v[0] = new SparseMatrix();
    for (int i=0; i<v1.length; i++) {
      //System.out.println(v1[i]);
      v[0].addEntry(i,v1[i]);
    }
    v[1] = new SparseMatrix();
    for (int i=0; i<v2.length; i++) {
      v[1].addEntry(i,v2[i]);
    }
    this.vecs = v;
  }
}


