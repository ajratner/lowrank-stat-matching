package lowrankparser;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.io.*;

import utils.SparseMatrix;

public class DepInstance implements Serializable {

	// coarse tag
    public static boolean loadCoarseTag = true;
	public static HashMap<String, String> coarseTag = null;

	public static void readCoarseMap(String file) throws IOException {
		String lang = file.substring(file.lastIndexOf("/") + 1);
		lang = lang.substring(0, lang.indexOf("."));
		String mapFile = file.substring(0, file.lastIndexOf("/") + 1) + lang
				+ ".uni.map";
		BufferedReader in = new BufferedReader(new FileReader(mapFile));
		String str = null;
		coarseTag = new HashMap<String, String>();
		while ((str = in.readLine()) != null) {
			String[] data = str.split("[ \t]+");
			coarseTag.put(data[0], data[1]);
		}
		in.close();
	}
	
	private static final long serialVersionUID = 1L;
	
	public int length;
	public String[] sentence;
    public String[] pos;
    public String[] posA;
    public String[] posC;
    public String[] labs;
    public int[] deps, labIds;
    //public transient FeatureVector fv;
    public String actParseTree;
    
    public SparseMatrix[] wordFvs;
    public SparseMatrix[][] bigramFvs;
    public SparseMatrix[][] arcFvs;
    public SparseMatrix[][][][] ntFvs;
    

    public DepInstance() {}
    
    public DepInstance(int length) { this.length = length; }
   
    public DepInstance(String[] sentence, String[] pos, String[] labs, int[] deps, int[] labIds) {
 		this.sentence = sentence;
 		this.pos = pos;
 		this.labs = labs;
 		this.deps = deps;
        this.labIds = labIds;
 		this.length = sentence.length;
 		createPosA();
 		createPosC();
     }
    
    private void createPosA() {
    	posA = new String[pos.length];
    	for (int i = 0; i < pos.length; i++)
    		posA[i] = pos[i].substring(0, 1);
    }
    
    private void createPosC() {
    	try {
	    	if (loadCoarseTag && coarseTag == null) {
                if (LowRankParser.trainFile != null)
	    		    readCoarseMap(LowRankParser.trainFile);
                else
                    readCoarseMap(LowRankParser.testFile);
	    	}
    	} catch (Exception e) {
    		//e.printStackTrace();
            System.out.println("Coarse POS tag mappings not found.");
            loadCoarseTag = false; 
    	}
        if (loadCoarseTag && coarseTag != null) {
            posC = new String[pos.length];
            for (int i = 0; i < pos.length; i++)
                posC[i] = coarseTag.containsKey(pos[i]) ? coarseTag.get(pos[i]) : pos[i];
        } else posC = posA;
    }
}
