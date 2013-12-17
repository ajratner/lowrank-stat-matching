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


public class DepPipe {
	
	public enum BigramType {
		MST, ALL
	}
    
    // "MST": use just mst bigram features
	  // "ALL": use all possible bigram combination
    public static BigramType bigramType = BigramType.ALL;
    public static boolean constructBigram = false;
    public static boolean constructSeenBigram = false;
    public static int bgHit = 0;
    public static int bgMiss = 0;
    
	public static boolean useLexicalFeature = true;
	
	public boolean labeled = false;				// whether the data file is labeled dependency file
	public static boolean learnLabel = false;	// whether learn labeled dependency or not
	public static int maxNumSent = -1;			// the maximum number of sentences to read & process

	
	public static HashMap<String, double[]> wordVectors = null;
	public static String unknowWord = "*UNKNOWN*";
	public static double[] unknownWv = null;
	public static boolean onlyLowerCase = false;
    public static int wvHit = 0;
    public static int wvMiss = 0;
    
	public Alphabet typeAlphabet;						// the alphabet of dependency labels
	public Alphabet wordAlphabet;						// the alphabet of word features
	public Alphabet arcAlphabet;						// the alphabet of arc features
    public Alphabet seenBigramAlphabet;	
    public boolean filterBigram;
	public String[] types;								// array that maps int index to dependency label string
	

	
	public static void loadWordVectors(String file) throws IOException {
		
		System.out.println("Loading word vectors...");
		
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(file),"UTF8"));
		wordVectors = new HashMap<String, double[]>();
		
		int upperCases = 0;
        int cnt = 0;
        double sumL2 = 0, minL2 = Double.POSITIVE_INFINITY, maxL2 = 0;
		String line = in.readLine();
		while (line != null) {
			line = line.trim();
			String[] parts = line.split("[ \t]");
			String word = parts[0];
			upperCases += Character.isUpperCase(word.charAt(0)) ? 1 : 0;
            ++cnt;
            double s = 0;
			double [] v = new double[parts.length - 1];
			for (int i = 0; i < v.length; ++i) {
				v[i] = Double.parseDouble(parts[i+1]);
                s += v[i]*v[i];
            }
			s = Math.sqrt(s);
            sumL2 += s;
            minL2 = Math.min(minL2, s);
            maxL2 = Math.max(maxL2, s);
			wordVectors.put(word, v);
			line = in.readLine();
		}
		in.close();

        sumL2 /= cnt;
        System.out.printf("Vector norm: Avg: %f  Min: %f  Max: %f%n", 
        		sumL2, minL2, maxL2);
        //for (double[] v : wordVectors.values())
        //	for (int i = 0; i < v.length; ++i) v[i] /= sumL2;

		if (wordVectors.containsKey(unknowWord))
			unknownWv = wordVectors.get(unknowWord);
		onlyLowerCase = upperCases < 100;
		System.out.printf("Num of word vectors: %d  Only lowercase:%b  %d%n", wordVectors.size(), onlyLowerCase, upperCases);
	}
	
	public DepPipe() {
		typeAlphabet = new Alphabet();
		wordAlphabet = new Alphabet();
		arcAlphabet = new Alphabet();
        seenBigramAlphabet = new Alphabet();
	}
	
	public void createAlphabets(String file) throws IOException {

		System.out.print("Creating Alphabet ... ");

		BufferedReader in =
		    new BufferedReader(new InputStreamReader(new FileInputStream(file),"UTF8"));
		String[][] lines = getLines(in);

		int cnt = 0;
			
		while(lines != null) {
				
		    String[] toks = lines[0];
		    String[] pos = lines[1];
		    String[] labs = lines[2];
		    String[] deps = lines[3];

		    for(int i = 0; i < labs.length; i++)
		    	typeAlphabet.lookupIndex(labs[i]);
				
		    int[] deps1 = new int[deps.length];
            int[] labs1 = new int[deps.length];
			for(int i = 0; i < deps.length; i++) {
				deps1[i] = Integer.parseInt(deps[i]);
                labs1[i] = typeAlphabet.lookupIndex(labs[i]);
            }
					    
			DepInstance pti = new DepInstance(toks,pos,labs,deps1,labs1);
		    
		    //createFeatures(pti);
		    initFeatureAlphabets(pti, deps1, labs);
				
		    lines = getLines(in);
		    cnt++;
	        if (maxNumSent != -1 && cnt >= maxNumSent) break;
		}
		
		closeAlphabets();

		in.close();

		System.out.println("Done.");
	}
	
    public void closeAlphabets() {
		
		typeAlphabet.stopGrowth();
		wordAlphabet.stopGrowth();
		arcAlphabet.stopGrowth();
        //seenBigramAlphabet.stopGrowth();

		types = new String[typeAlphabet.size()];
		Object[] keys = typeAlphabet.toArray();
		for(int i = 0; i < keys.length; i++) {
		    int indx = typeAlphabet.lookupIndex(keys[i]);
		    types[indx] = (String)keys[i];
		}
	
    }
    



    public DepInstance[] createInstances(String file) throws IOException {

		BufferedReader in = //new BufferedReader(new FileReader(file));
		new BufferedReader(new InputStreamReader(new FileInputStream(file),"UTF8"));
		String[][] lines = getLines(in);
		
		LinkedList<DepInstance> lt = new LinkedList<DepInstance>();
				
		int num1 = 0;
		while(lines != null) {
			if ((num1+1) % 100 == 0)
				System.out.printf("Creating Dependency Instance: %d%n", num1+1);
			
			String[] toks = lines[0];
			String[] pos = lines[1];
			String[] labs = lines[2];
			String[] deps = lines[3];
			
			int[] deps1 = new int[deps.length];
            int[] labs1 = new int[deps.length];
			for(int i = 0; i < deps.length; i++) {
				deps1[i] = Integer.parseInt(deps[i]);
                labs1[i] = typeAlphabet.lookupIndex(labs[i]);
            }
					    
			DepInstance pti = new DepInstance(toks,pos,labs,deps1,labs1);
		    
			String spans = "";
		    for(int i = 1; i < deps.length; i++) {
		    	spans += deps[i]+"|"+i+":"+typeAlphabet.lookupIndex(labs[i])+" ";
		    }		
		    pti.actParseTree = spans.trim();
			
		    createFeatures(pti);
			lt.add(pti);
			
			lines = getLines(in);
			num1++;
			if (maxNumSent != -1 && num1 >= maxNumSent) break;
		}
		System.out.printf("%d instances in total.%n%n", num1);
		closeAlphabets();
        seenBigramAlphabet.stopGrowth();
		in.close();
		
		DepInstance[] pti = new DepInstance[lt.size()];
		int N = 0;
		for (DepInstance inst : lt) {
			pti[N++] = inst;
		}
		
		return pti;
		
	}
    
    public DepInstance createInstance(BufferedReader in) throws IOException {
    	String[][] lines = getLines(in);
    	if (lines == null) return null;
    	
    	String[] toks = lines[0];
		String[] pos = lines[1];
		String[] labs = lines[2];
		String[] deps = lines[3];
		
		int[] deps1 = new int[deps.length];
            int[] labs1 = new int[deps.length];
			for(int i = 0; i < deps.length; i++) {
				deps1[i] = Integer.parseInt(deps[i]);
                labs1[i] = typeAlphabet.lookupIndex(labs[i]);
            }
					    
		DepInstance pti = new DepInstance(toks,pos,labs,deps1,labs1);
		    
		String spans = "";
	    for(int i = 1; i < deps.length; i++) {
	    	spans += deps[i]+"|"+i+":"+typeAlphabet.lookupIndex(labs[i])+" ";
	    }		
	    pti.actParseTree = spans.trim();
		
	    createFeatures(pti);
	    
	    return pti;
    }
    
    public void setLabeled(String file) throws IOException {
		BufferedReader in = new BufferedReader(new FileReader(file));
		in.readLine(); in.readLine(); in.readLine();
		String line = in.readLine();
		if(line.trim().length() > 0) labeled = true;
		in.close();
    }
    
    public String[][] getLines(BufferedReader in) throws IOException {
    	String line = in.readLine();
    	String pos_line = in.readLine();
    	String lab_line = labeled ? in.readLine() : pos_line;
    	String deps_line = in.readLine();
    	in.readLine(); // blank line

    	if(line == null) return null;

    	String[] toks = line.split("\t");
    	String[] pos = pos_line.split("\t");
    	String[] labs = lab_line.split("\t");
    	String[] deps = deps_line.split("\t");
    	
    	String[] toks_new = new String[toks.length+1];
    	String[] pos_new = new String[pos.length+1];
    	String[] labs_new = new String[labs.length+1];
    	String[] deps_new = new String[deps.length+1];
    	toks_new[0] = "<root>";
    	pos_new[0] = "<root-POS>";
    	labs_new[0] = "<no-type>";
    	deps_new[0] = "-1";
    	for(int i = 0; i < toks.length; i++) {
    	    toks_new[i+1] = normalize(toks[i]);
    	    pos_new[i+1] = pos[i];
    	    labs_new[i+1] = (labeled && learnLabel) ? labs[i] : "<no-type>";
    	    deps_new[i+1] = deps[i];
    	}
    	toks = toks_new;
    	pos = pos_new;
    	labs = labs_new;
    	deps = deps_new;
    	
    	String[][] result = new String[4][];
    	result[0] = toks; result[1] = pos; result[2] = labs; result[3] = deps;
    	return result;
    }
    

    
  public String normalize(String s) {
  if(s.matches("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+"))
      return "<num>";
  return s;
  }
    
    public void initFeatureAlphabets(DepInstance inst, int[] deps, String[] labs) {
        
        //SparseMatrix[] wordFvs = new SparseMatrix[inst.length];
        for (int i = 0; i < inst.length; ++i)
            createFeatures(inst, i);
            //wordFvs[i] = createFeatures(inst, i);

    	for (int i = 0; i < inst.length; ++i) {
    		
    		if (deps[i] == -1) continue;
    	     
    		int parent = deps[i];
    		createFeatures(inst, parent, i);	// arc features
    		if (labeled && learnLabel) {
    			String type = labs[i]; 
    			boolean toRight = parent < i;
    			createFeatures(inst, parent, type, toRight, false);
    			createFeatures(inst, i, type, toRight, true);
    		}
    	}

    }
    
    public void createFeatures(DepInstance inst) {
    	
    	// create features for each word in the sentence
    	SparseMatrix[] wordFvs = new SparseMatrix[inst.length];
    	for (int i = 0; i < inst.length; ++i) {
    		wordFvs[i] = createFeatures(inst, i);
    	}
    	inst.wordFvs = wordFvs;
        
        if (constructBigram) {
    		SparseMatrix[][] bigramFvs = new SparseMatrix[inst.length][inst.length];
    		for (int i = 0; i < inst.length; ++i)
    			for (int j = 0; j < inst.length; ++j) if (i != j) {
    				SparseMatrix mat;
    				if (bigramType == BigramType.MST)
    					mat = createMstBigramFeatures(inst, i, j);
    				else
    					mat = SparseMatrix.outerProduct(wordFvs[i], wordFvs[j]);
                    
                    if (constructSeenBigram && filterBigram) {
                        for (int k = 0, K = mat.size(); k < K; ++k) {
                            int x = mat.x(k), y = mat.y(k);
                            if (seenBigramAlphabet.lookupIndex(x+" "+y) < 0) {
                                mat.setZ(k, 0.0);
                                bgMiss++;
                            } else bgHit++;
                        } 
                    }

                    if (constructSeenBigram && !seenBigramAlphabet.growthStopped()) {
                        for (int k = 0, K = mat.size(); k < K; ++k) {
                            int x = mat.x(k), y = mat.y(k);
                            seenBigramAlphabet.lookupIndex(x+" "+y);
                        }
                    }

                    bigramFvs[i][j] = mat;
    			}
    		inst.bigramFvs = bigramFvs;
    	}
    	
    	// create features for each arc (i->j) in the sentence
    	SparseMatrix[][] arcFvs = new SparseMatrix[inst.length][inst.length];
    	for (int i = 0; i < inst.length; ++i) 
    		for (int j = i + 1; j < inst.length; ++j) {
    			arcFvs[i][j] = createFeatures(inst, i, j);
    			arcFvs[j][i] = createFeatures(inst, j, i);
    		}
    	inst.arcFvs = arcFvs;
    	
    	// create label features for arc (i--type-->) or (--type-->i) in the sentence
    	if (labeled && learnLabel) {
	    	SparseMatrix[][][][] ntFvs = new SparseMatrix[inst.length][types.length][2][2];
	    	for (int i = 0; i < inst.length; ++i)
	    		for (int t = 0; t < types.length; ++t) 
	    			for (int j = 0; j < 2; ++j)
	    				for (int k =0; k < 2; ++k) {
	    					ntFvs[i][t][j][k] = createFeatures(inst, i, types[t], j == 1, k == 1);
	    				}
	    	inst.ntFvs = ntFvs;
    	}
    }
    
    public SparseMatrix createFeatures(DepInstance inst, int word,
    		String type, boolean toRight, boolean isChild) {
    	
    	SparseMatrix mat = new SparseMatrix(arcAlphabet.size());
    	if (!labeled) return mat;
    	
    	String att = "";
    	if(toRight)
    	    att = "RA";
    	else
    	    att = "LA";
    	att+="&"+isChild;
    	
    	String[] toks = inst.sentence;
    	String[] pos = inst.pos;
    	
    	String w = toks[word];
    	String wP = pos[word];

    	String wPm1 = word > 0 ? pos[word-1] : "STR";
    	String wPp1 = word < pos.length-1 ? pos[word+1] : "END";

    	addArcFeature("NTS1="+type+"&"+att, 1.0, mat);
    	addArcFeature("ANTS1="+type, 1.0, mat);
    	for (int i = 0; i < 2; i++) {
    	    String suff = i < 1 ? "&"+att : "";
    	    suff = "&"+type+suff;

    	    addArcFeature("NTH="+w+" "+wP+suff, 1.0, mat);
    	    addArcFeature("NTI="+wP+suff, 1.0, mat);
    	    addArcFeature("NTIA="+wPm1+" "+wP+suff, 1.0, mat);
    	    addArcFeature("NTIB="+wP+" "+wPp1+suff, 1.0, mat);
    	    addArcFeature("NTIC="+wPm1+" "+wP+" "+wPp1+suff, 1.0, mat);
    	    addArcFeature("NTJ="+w+suff, 1.0, mat); 
    			
    	}
    	
    	return mat;
    }
    
    
    public SparseMatrix createFeatures(DepInstance inst, int h, int c) {
    	
    	String att = "";
    	if(h < c)
    	    att = "RA";
    	else
    	    att = "LA";
    	
    	int dist = Math.abs(h-c);
    	String distBool = "0";
    	if(dist > 1)
    	    distBool = "1";
    	if(dist > 2)
    	    distBool = "2";
    	if(dist > 3)
    	    distBool = "3";
    	if(dist > 4)
    	    distBool = "4";
    	if(dist > 5)
    	    distBool = "5";
    	if(dist > 10)
    	    distBool = "10";
    		
    	String attDist = "&"+att+"&"+distBool;
    	
    	String[] pos = inst.pos;
    	String[] posA = inst.posA;
    	String[] toks = inst.sentence;
    	
    	SparseMatrix mat = new SparseMatrix(arcAlphabet.size());
    	
    	//addArcFeature("Dist=" + attDist, 1.0, mat);
    	
    	
    	/*****************************************************
    	 * features defined in MST parser
    	 *****************************************************/    	
    	String head = toks[h], child = toks[c];
    	String headP = pos[h], childP = pos[c];
    	
    	String all = head + " " + headP + " " + child + " " + childP;
    	String hPos = headP + " " + child + " " + childP;
    	String cPos = head + " " + headP + " " + childP;
    	String hP = headP + " " + child;
    	String cP = head + " " + childP;
    	String oPos = headP + " " + childP;
    	String oLex = head + " " + child;
    	
    	addArcFeature("A="+all+attDist,1.0,mat); 
    	addArcFeature("B="+hPos+attDist,1.0,mat);
    	addArcFeature("C="+cPos+attDist,1.0,mat);
    	addArcFeature("D="+hP+attDist,1.0,mat);
    	addArcFeature("E="+cP+attDist,1.0,mat);
    	addArcFeature("F="+oLex+attDist,1.0,mat);  	
    	addArcFeature("H="+head+" "+headP+attDist,1.0,mat);    	
    	addArcFeature("J="+head+attDist,1.0,mat); 
    	addArcFeature("K="+child+" "+childP+attDist,1.0,mat);    	
    	addArcFeature("M="+child+attDist,1.0,mat);     	
    	addArcFeature("G="+oPos+attDist,1.0,mat);
    	addArcFeature("I="+headP+attDist,1.0,mat);
    	addArcFeature("L="+childP+attDist,1.0,mat);
    	
    	addArcFeature("A1="+all,1.0,mat); 
    	addArcFeature("B1="+hPos,1.0,mat);
    	addArcFeature("C1="+cPos,1.0,mat);
    	addArcFeature("D1="+hP,1.0,mat);
    	addArcFeature("E1="+cP,1.0,mat);
    	addArcFeature("F1="+oLex,1.0,mat);  	
    	addArcFeature("H1="+head+" "+headP,1.0,mat);    	
    	addArcFeature("J1="+head,1.0,mat); 
    	addArcFeature("K1="+child+" "+childP,1.0,mat);    	
    	addArcFeature("M1="+child,1.0,mat);     	
    	addArcFeature("G1="+oPos,1.0,mat);
    	addArcFeature("I1="+headP,1.0,mat);
    	addArcFeature("L1="+childP,1.0,mat);
    	
    	int small = h < c ? h : c;
    	int large = h < c ? c : h;
    	String pLeft = small > 0 ? pos[small-1] : "STR";
    	String pRight = large < pos.length-1 ? pos[large+1] : "END";
    	String pLeftRight = small < large-1 ? pos[small+1] : "MID";
    	String pRightLeft = large > small+1 ? pos[large-1] : "MID";
    	String pLeftA = small > 0 ? posA[small-1] : "STR";
    	String pRightA = large < pos.length-1 ? posA[large+1] : "END";
    	String pLeftRightA = small < large-1 ? posA[small+1] : "MID";
    	String pRightLeftA = large > small+1 ? posA[large-1] : "MID";
    	
    	// feature posR posMid posL
    	for(int i = small+1; i < large; i++) {
    	    String allPos = pos[small]+" "+pos[i]+" "+pos[large];
    	    String allPosA = posA[small]+" "+posA[i]+" "+posA[large];
    	    addArcFeature("PC="+allPos+attDist,1.0,mat);
    	    addArcFeature("1PC="+allPos,1.0,mat);
    	    addArcFeature("XPC="+allPosA+attDist,1.0,mat);
    	    addArcFeature("X1PC="+allPosA,1.0,mat);
    	}
    	
    	// feature posL-1 posL posR posR+1
    	addArcFeature("PT="+pLeft+" "+pos[small]+" "+pos[large]+" "+pRight+attDist,1.0,mat);
    	addArcFeature("PT1="+pos[small]+" "+pos[large]+" " +pRight+attDist,1.0,mat);
    	addArcFeature("PT2="+pLeft+" "+pos[small]+" "+pos[large]+attDist,1.0,mat);
    	addArcFeature("PT3="+pLeft+" "+pos[large]+" "+pRight+attDist,1.0,mat);
    	addArcFeature("PT4="+pLeft+" "+pos[small]+" "+pRight+attDist,1.0,mat);
    	
    	addArcFeature("XPT="+pLeftA+" "+posA[small]+" "+posA[large]+" "+pRightA+attDist,1.0,mat);
    	addArcFeature("XPT1="+posA[small]+" "+posA[large]+" " +pRightA+attDist,1.0,mat);
    	addArcFeature("XPT2="+pLeftA+" "+posA[small]+" "+posA[large]+attDist,1.0,mat);
    	addArcFeature("XPT3="+pLeftA+" "+posA[large]+" "+pRightA+attDist,1.0,mat);
    	addArcFeature("XPT4="+pLeftA+" "+posA[small]+" "+pRightA+attDist,1.0,mat);
    	
    	// feature posL posL+1 posR-1 posR
    	addArcFeature("APT="+pos[small]+" "+pLeftRight+" "
    		 +pRightLeft+" "+pos[large]+attDist,1.0,mat);
    	addArcFeature("APT1="+pos[small]+" "+pRightLeft+" "+pos[large]+attDist,1.0,mat);
    	addArcFeature("APT2="+pos[small]+" "+pLeftRight+" "+pos[large]+attDist,1.0,mat);
    	addArcFeature("APT3="+pLeftRight+" "+pRightLeft+" "+pos[large]+attDist,1.0,mat);
    	addArcFeature("APT4="+pos[small]+" "+pLeftRight+" "+pRightLeft+attDist,1.0,mat);
    	
    	addArcFeature("XAPT="+posA[small]+" "+pLeftRightA+" "
    			 +pRightLeftA+" "+posA[large]+attDist,1.0,mat);
		addArcFeature("XAPT1="+posA[small]+" "+pRightLeftA+" "+posA[large]+attDist,1.0,mat);
		addArcFeature("XAPT2="+posA[small]+" "+pLeftRightA+" "+posA[large]+attDist,1.0,mat);
		addArcFeature("XAPT3="+pLeftRightA+" "+pRightLeftA+" "+posA[large]+attDist,1.0,mat);
		addArcFeature("XAPT4="+posA[small]+" "+pLeftRightA+" "+pRightLeftA+attDist,1.0,mat);
    	
		// feature posL-1 posL posR-1 posR
		// feature posL posL+1 posR posR+1
		addArcFeature("BPT="+pLeft+" "+pos[small]+" "+pRightLeft+" "+pos[large]+attDist,1.0,mat);
		addArcFeature("CPT="+pos[small]+" "+pLeftRight+" "+pos[large]+" "+pRight+attDist,1.0,mat);
			
		addArcFeature("XBPT="+pLeftA+" "+posA[small]+" "+pRightLeftA+" "+posA[large]+attDist,1.0,mat);
		addArcFeature("XCPT="+posA[small]+" "+pLeftRightA+" "+posA[large]+" "+pRightA+attDist,1.0,mat);
    	
		
    	// feature posL-1 posL posR posR+1
    	addArcFeature("PT1="+pLeft+" "+pos[small]+" "+pos[large]+" "+pRight,1.0,mat);
    	addArcFeature("PT11="+pos[small]+" "+pos[large]+" " +pRight,1.0,mat);
    	addArcFeature("PT21="+pLeft+" "+pos[small]+" "+pos[large],1.0,mat);
    	addArcFeature("PT31="+pLeft+" "+pos[large]+" "+pRight,1.0,mat);
    	addArcFeature("PT41="+pLeft+" "+pos[small]+" "+pRight,1.0,mat);
    	
    	addArcFeature("XPT1="+pLeftA+" "+posA[small]+" "+posA[large]+" "+pRightA,1.0,mat);
    	addArcFeature("XPT11="+posA[small]+" "+posA[large]+" " +pRightA,1.0,mat);
    	addArcFeature("XPT21="+pLeftA+" "+posA[small]+" "+posA[large],1.0,mat);
    	addArcFeature("XPT31="+pLeftA+" "+posA[large]+" "+pRightA,1.0,mat);
    	addArcFeature("XPT41="+pLeftA+" "+posA[small]+" "+pRightA,1.0,mat);
    	
    	// feature posL posL+1 posR-1 posR
    	addArcFeature("APT1="+pos[small]+" "+pLeftRight+" "
    		 +pRightLeft+" "+pos[large],1.0,mat);
    	addArcFeature("APT11="+pos[small]+" "+pRightLeft+" "+pos[large],1.0,mat);
    	addArcFeature("APT21="+pos[small]+" "+pLeftRight+" "+pos[large],1.0,mat);
    	addArcFeature("APT31="+pLeftRight+" "+pRightLeft+" "+pos[large],1.0,mat);
    	addArcFeature("APT41="+pos[small]+" "+pLeftRight+" "+pRightLeft,1.0,mat);
    	
    	addArcFeature("XAPT1="+posA[small]+" "+pLeftRightA+" "
    			 +pRightLeftA+" "+posA[large],1.0,mat);
		addArcFeature("XAPT11="+posA[small]+" "+pRightLeftA+" "+posA[large],1.0,mat);
		addArcFeature("XAPT21="+posA[small]+" "+pLeftRightA+" "+posA[large],1.0,mat);
		addArcFeature("XAPT31="+pLeftRightA+" "+pRightLeftA+" "+posA[large],1.0,mat);
		addArcFeature("XAPT41="+posA[small]+" "+pLeftRightA+" "+pRightLeftA,1.0,mat);
    	
		// feature posL-1 posL posR-1 posR
		// feature posL posL+1 posR posR+1
		addArcFeature("BPT1="+pLeft+" "+pos[small]+" "+pRightLeft+" "+pos[large],1.0,mat);
		addArcFeature("CPT1="+pos[small]+" "+pLeftRight+" "+pos[large]+" "+pRight,1.0,mat);
			
		addArcFeature("XBPT1="+pLeftA+" "+posA[small]+" "+pRightLeftA+" "+posA[large],1.0,mat);
		addArcFeature("XCPT1="+posA[small]+" "+pLeftRightA+" "+posA[large]+" "+pRightA,1.0,mat);
		
		
		if(head.length() > 5 || child.length() > 5) {
		    int hL = head.length();
		    int cL = child.length();
			    
		    head = hL > 5 ? head.substring(0,5) : head;
		    child = cL > 5 ? child.substring(0,5) : child;
			    
		    all = head + " " + headP + " " + child + " " + childP;
		    hPos = headP + " " + child + " " + childP;
		    cPos = head + " " + headP + " " + childP;
		    hP = headP + " " + child;
		    cP = head + " " + childP;
		    oPos = headP + " " + childP;
		    oLex = head + " " + child;
		
		    addArcFeature("SA="+all+attDist,1.0,mat);
		    addArcFeature("SF="+oLex+attDist,1.0,mat); //this
		    addArcFeature("SAA="+all,1.0,mat); //this
		    addArcFeature("SFF="+oLex,1.0,mat); //this

		    if(cL > 5) {
				addArcFeature("SB="+hPos+attDist,1.0,mat);
				addArcFeature("SD="+hP+attDist,1.0,mat);
				addArcFeature("SK="+child+" "+childP+attDist,1.0,mat);
				addArcFeature("SM="+child+attDist,1.0,mat); //this
				addArcFeature("SBB="+hPos,1.0,mat);
				addArcFeature("SDD="+hP,1.0,mat);
				addArcFeature("SKK="+child+" "+childP,1.0,mat);
				addArcFeature("SMM="+child,1.0,mat); //this
		    }
		    if(hL > 5) {
				addArcFeature("SC="+cPos+attDist,1.0,mat);
				addArcFeature("SE="+cP+attDist,1.0,mat);
				addArcFeature("SH="+head+" "+headP+attDist,1.0,mat);
				addArcFeature("SJ="+head+attDist,1.0,mat); //this
					
				addArcFeature("SCC="+cPos,1.0,mat);
				addArcFeature("SEE="+cP,1.0,mat);
				addArcFeature("SHH="+head+" "+headP,1.0,mat);
				addArcFeature("SJJ="+head,1.0,mat); //this
		    }
		}		
		
    	return mat;
    }
        
    public SparseMatrix createFeatures(DepInstance inst, int i) {
    	
    	String[] pos = inst.pos;
        String[] posA = inst.posA;
        String[] posC = inst.posC;
        String[] toks = inst.sentence;
        
        String w0 = toks[i];
        String p0 = pos[i];
        String pa0 = posA[i];
        String pc0 = posC[i];
    	String pLeft = i > 0 ? pos[i-1] : "STR";
    	String pRight = i < pos.length-1 ? pos[i+1] : "END";
    	String pLeftA = i > 0 ? posA[i-1] : "STR";
    	String pRightA = i < pos.length-1 ? posA[i+1] : "END";
    	String pLeftC = i > 0 ? posC[i-1] : "STR";
    	String pRightC = i < pos.length-1 ? posC[i+1] : "END";
    	
    	SparseMatrix mat = new SparseMatrix(wordAlphabet.size());
        
        addWordFeature("1", 1.0, mat);  // add "1" in order to include unigram features

    	if (useLexicalFeature) {
    		addWordFeature("W0=" + w0, 1.0, mat);
    		addWordFeature("W0P0=" + w0 + " " + p0, 1.0, mat); 
    	}    	
    	addWordFeature("P0=" + p0, 1.0, mat);    	
    	addWordFeature("P-1=" + pLeft, 1.0, mat);
    	addWordFeature("P1=" + pRight, 1.0, mat);
    	   	
    	addWordFeature("P-1P0=" + pLeft + " " + p0, 1.0, mat);
    	addWordFeature("P0P1=" + p0 + " " + pRight, 1.0, mat);
    	
    	addWordFeature("PA-1=" + pLeftA, 1.0, mat);
    	addWordFeature("PA1=" + pRightA, 1.0, mat);
    	addWordFeature("PA0=" + pa0, 1.0, mat);
    	addWordFeature("PA-1PA0=" + pLeftA + " " + pa0, 1.0, mat);    	
    	addWordFeature("PA0PA1=" + pa0 + " " + pRightA, 1.0, mat);	
    	
    	if (DepInstance.coarseTag != null) {
    		addWordFeature("PC-1=" + pLeftC, 1.0, mat);
    		addWordFeature("PC1=" + pRightC, 1.0, mat);
    		addWordFeature("PC0=" + pc0, 1.0, mat);
        	addWordFeature("PC-1PC0=" + pLeftC + " " + pc0, 1.0, mat);
        	addWordFeature("PC0PC1=" + pc0 + " " + pRightC, 1.0, mat);
    	}
    	
    	if (wordVectors != null) {
    		double [] v = unknownWv;
    		String ww = onlyLowerCase ? w0.toLowerCase() : w0;
    		if (wordVectors.containsKey(ww)) {
    			v = wordVectors.get(ww);
                wvHit++;
            } else if (!ww.equals("<root>") && !ww.equals("<num>")) {
                wvMiss++;
                //System.out.printf("%s\t", ww);
            }
    		if (v != null) {
    			for (int j = 0; j < v.length; ++j)
    				addWordFeature("WV[" + j + "]", v[j], mat);
    		}
    	}
    	
    	return mat;
    }
    
    public SparseMatrix createMstBigramFeatures(DepInstance inst, int head, int child) {
        String[] pos = inst.pos;
        String[] posA = inst.posA;
        String[] toks = inst.sentence;
        
        String wh = toks[head];
        String ph = pos[head];
        String phA = posA[head];
    	String phLeft = head > 0 ? pos[head-1] : "STR";
    	String phRight = head < pos.length-1 ? pos[head+1] : "END";
        String phLeftA = head > 0 ? posA[head-1] : "STR";
    	String phRightA = head < pos.length-1 ? posA[head+1] : "END";
    	
        String wc = toks[child];
        String pc = pos[child];
        String pcA = posA[child];
        String pcLeft = child > 0 ? pos[child-1] : "STR";
    	String pcRight = child < pos.length-1 ? pos[child+1] : "END";
        String pcLeftA = child > 0 ? posA[child-1] : "STR";
    	String pcRightA = child < pos.length-1 ? posA[child+1] : "END";
    	
    	SparseMatrix mat = new SparseMatrix(wordAlphabet.size(), wordAlphabet.size());
        
        addBigramFeature("P-1P0="+phLeft+" "+ph, "P0P1="+pc+" "+pcRight, 1.0, mat);
        addBigramFeature("P0P1="+ph+" "+phRight, "P-1P0="+pcLeft+" "+pc, 1.0, mat);
    	addBigramFeature("P-1P0="+phLeft+" "+ph, "P0="+pc, 1.0, mat);
        addBigramFeature("P0="+ph, "P-1P0="+pcLeft+" "+pc, 1.0, mat);
        addBigramFeature("P-1P0="+phLeft+" "+ph, "P1="+pcRight, 1.0, mat);
        addBigramFeature("P1="+phRight, "P-1P0="+pcLeft+" "+pc, 1.0, mat);
        addBigramFeature("P0="+ph, "P0P1="+pc+" "+pcRight, 1.0, mat);
        addBigramFeature("P0P1="+ph+" "+phRight, "P0="+pc, 1.0, mat);
        addBigramFeature("P-1="+phLeft, "P0P1="+pc+" "+pcRight, 1.0, mat);
        addBigramFeature("P0P1="+ph+" "+phRight, "P-1="+pcLeft, 1.0, mat);
        addBigramFeature("P-1P0="+phLeft+" "+ph, "P-1P0="+pcLeft+" "+pc, 1.0, mat);
        addBigramFeature("P0P1="+ph+" "+phRight, "P0P1="+pc+" "+pcRight, 1.0, mat);

        addBigramFeature("PA-1PA0="+phLeftA+" "+phA, "PA0PA1="+pcA+" "+pcRightA, 1.0, mat);
        addBigramFeature("PA0PA1="+phA+" "+phRightA, "PA-1PA0="+pcLeftA+" "+pcA, 1.0, mat);
    	addBigramFeature("PA-1PA0="+phLeftA+" "+phA, "PA0="+pcA, 1.0, mat);
        addBigramFeature("PA0="+phA, "PA-1PA0="+pcLeftA+" "+pcA, 1.0, mat);
        addBigramFeature("PA-1PA0="+phLeftA+" "+phA, "PA1="+pcRightA, 1.0, mat);
        addBigramFeature("PA1="+phRightA, "PA-1PA0="+pcLeftA+" "+pcA, 1.0, mat);
        addBigramFeature("PA0="+phA, "PA0PA1="+pcA+" "+pcRightA, 1.0, mat);
        addBigramFeature("PA0PA1="+phA+" "+phRightA, "PA0="+pcA, 1.0, mat);
        addBigramFeature("PA-1="+phLeftA, "PA0PA1="+pcA+" "+pcRightA, 1.0, mat);
        addBigramFeature("PA0PA1="+phA+" "+phRightA, "PA-1="+pcLeftA, 1.0, mat);
        addBigramFeature("PA-1PA0="+phLeftA+" "+phA, "PA-1PA0="+pcLeftA+" "+pcA, 1.0, mat);
        addBigramFeature("PA0PA1="+phA+" "+phRightA, "PA0PA1="+pcA+" "+pcRightA, 1.0, mat);

        addBigramFeature("W0P0="+wh+" "+ph, "W0P0="+wc+" "+pc, 1.0, mat);
        addBigramFeature("P0="+ph, "W0P0="+wc+" "+pc, 1.0, mat);
        addBigramFeature("W0P0="+wh+" "+ph, "P0="+pc, 1.0, mat);
        addBigramFeature("P0="+ph, "W0="+wc, 1.0, mat);
        addBigramFeature("W0="+wh, "P0="+pc, 1.0, mat);
        addBigramFeature("W0="+wh, "W0="+wc, 1.0, mat);
        addBigramFeature("P0="+ph, "P0="+pc, 1.0, mat);
        addBigramFeature("W0P0="+wh+" "+ph, "1", 1.0, mat);
        addBigramFeature("W0="+wh, "1", 1.0, mat);
        addBigramFeature("P0="+ph, "1", 1.0, mat);
        addBigramFeature("1", "W0P0="+wc+" "+pc, 1.0, mat);
        addBigramFeature("1", "W0="+wc, 1.0, mat);
        addBigramFeature("1", "P0="+pc, 1.0, mat);
        
        return mat;
    }
    
    public void addBigramFeature(String h, String c, double value, SparseMatrix mat) {
        int ih = wordAlphabet.lookupIndex(h);
        int ic = wordAlphabet.lookupIndex(c);
        if (ih >= 0 && ic >= 0)
            mat.addEntry(ih, ic, value);
    }

    public void addWordFeature(String feat, double value, SparseMatrix mat) {
    	int id = wordAlphabet.lookupIndex(feat);
    	if (id >= 0)
    		mat.addEntry(id, value);
    }

    public void addArcFeature(String feat, double value, SparseMatrix mat) {
    	int id = arcAlphabet.lookupIndex(feat);
    	if (id >= 0)
    		mat.addEntry(id, value);
    }
    
}
