/*
 *   This DependencyPipe.java is derive from MSTParser code 
 */

package lowrankparser;


import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.LinkedList;

import utils.SparseMatrix;


// A: barebone class
public class PairPipe {

  public static int featureLength;

  // A: we will just load the feature vectors directly, as printed from python code
  // format = vec1 // vec2 // d_vec // label (0/1) // //
  public PairInstance[] createInstance(String file) throws IOException {

  BufferedReader in = //new BufferedReader(new FileReader(file));
  new BufferedReader(new InputStreamReader(new FileInputStream(file),"UTF8"));
  String[][] lines = getLines(in);
  
  LinkedList<PairInstance> lt = new LinkedList<PairInstance>();
      
  int num1 = 0;
  while(lines != null) {
    if ((num1+1) % 10000 == 0)
      System.out.printf("Creating Pair Instance: %d%n", num1+1);
    
    String[] vec1 = lines[0];
    String[] vec2 = lines[1];
    String[] label = lines[2];
    
    float[] v1 = new float[vec1.length];
    float[] v2 = new float[vec2.length];
    boolean lbl = false;

    this.featureLength = v1.length;

    for (int i=0; i<vec1.length; i++) {
      //System.out.println(vec1[i]);
      v1[i] = Float.parseFloat(vec1[i]);
    }
    for (int i=0; i<vec2.length; i++) {
      v2[i] = Float.parseFloat(vec2[i]);
    }
    if (label[0].equals("1.0")) {
      lbl = true;
    } else {
      lbl = false;
    }

    PairInstance pti = new PairInstance(v1,v2,lbl);
    lt.add(pti);
    
    lines = getLines(in);
    num1++;
  }

  PairInstance[] pti = new PairInstance[lt.size()];
  int N = 0;
  for (PairInstance inst : lt) {
    pti[N++] = inst;
  }
  
  return pti;
  }

  public PairInstance createInstance(BufferedReader in) throws IOException {
    	String[][] lines = getLines(in);
    	if (lines == null) return null;
    	
    String[] vec1 = lines[0];
    String[] vec2 = lines[1];
    String[] label = lines[2];
    
    float[] v1 = new float[vec1.length];
    float[] v2 = new float[vec2.length];
    boolean lbl = false;
					    
		PairInstance pti = new PairInstance(v1,v2,lbl);
		
    /*    
		String spans = "";
	    for(int i = 1; i < deps.length; i++) {
	    	spans += deps[i]+"|"+i+":"+typeAlphabet.lookupIndex(labs[i])+" ";
	    }		
	    pti.actParseTree = spans.trim();
		
	    createFeatures(pti);
    */
	    
	    return pti;
    }

  
  // A: for our format...
  public String[][] getLines(BufferedReader in) throws IOException {
    String v1_line = in.readLine();
    String v2_line = in.readLine();
    String lbl_line = in.readLine();
    in.readLine(); // blank line

    if (v1_line == null) return null;

    String[] v1 = v1_line.split("\t");
    String[] v2 = v2_line.split("\t");
    String[] lbl = lbl_line.split("\t");
    
    
    String[][] result = new String[3][];
    result[0] = v1; result[1] = v2; result[2] = lbl;
    return result;
  }


}
