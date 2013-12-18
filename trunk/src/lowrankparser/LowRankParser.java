package lowrankparser;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.Random;

import utils.LowRankMatrix;

public class LowRankParser {

	public static String trainFile = null;
	public static String testFile = null;
	public static String outFile = null;
	public static boolean train = false;
	public static boolean test = false;	
	public static String wordVectorFile = null;
	public static String modelFile = "model.out";
    public static boolean evalWithPunc = true;

	public static int maxNumIters = 100;
  public static double bestDevUAS = -1; 
	//public static DepPipe pipe;
	public static PairPipe pipe;
	public static LowRankModel model;
	
	public static int batchSize = 10;
	public static int numThreads = 4;
	public static ConcurrentLinkedQueue<DecodingInstance> taskQueue 
        = new ConcurrentLinkedQueue<DecodingInstance>();	
	
    public static void main (String[] args) throws Exception {
    	
		processArguments(args);
		
		pipe = null;
		model = null;
		//if (wordVectorFile != null) 
		//	PairPipe.loadWordVectors(wordVectorFile);
		
		if (train) {
			pipe = new PairPipe();
      
      //pipe.setLabeled(trainFile);
			//pipe.createAlphabets(trainFile);
			//System.out.printf("Num of Features: %d %d%n", 
      //              pipe.wordAlphabet.size(), 
      //              pipe.arcAlphabet.size());
      //      System.out.printf("Num of labels: %d%n", pipe.types.length);
      //      System.out.printf("WV unseen rate: %f%n", 
      //              (DepPipe.wvMiss + 1e-8)/(DepPipe.wvMiss + DepPipe.wvHit + 1e-8));
      //      System.out.println();
			
			//DepInstance[] lstTrain = pipe.createInstances(trainFile);
      //      if (DepPipe.constructBigram && DepPipe.constructSeenBigram)
      //          System.out.printf("Num of Seen Bigram Features: %d%n",
      //              pipe.seenBigramAlphabet.size());

      // A: we will just load the feature vectors directly, as printed from python code
      // format = vec1 // vec2 // d_vec // label (0/1) // //
      PairInstance[] lstTrain = pipe.createInstance(trainFile);

			model = new LowRankModel(pipe, lstTrain.length);
			
      //System.out.println(pipe.featureLength);
      
      train(lstTrain, model, pipe);
			
      // A: testing stop
      System.exit(0);

			if (bestDevUAS == -1) saveModel();
		}
		
		if (test) {
			
			//if (!train) {
                System.out.println("Loading model...");
				loadModel();
		    	//pipe.setLabeled(testFile);
			//}
			
			System.out.println("_____________________________________________");
			System.out.println();
			System.out.printf(" Evaluation: %s%n", testFile);
			System.out.println();
			evaluateSet(model, pipe, true);
            if (!train) {           // also see UAS when seen features are removed
                /*
                if (pipe.constructSeenBigram) {
                    pipe.filterBigram = true;
                    evaluateSet(model, pipe);
                    System.out.printf(" Bigram: %d %f (%d/%d)%n",
                            pipe.seenBigramAlphabet.size(),
                            pipe.bgHit/(pipe.bgHit+pipe.bgMiss+1e-30),
                            pipe.bgHit, pipe.bgMiss);
                    pipe.filterBigram = false;
                }*/
                evalWithPunc = false;
                evaluateSet(model, pipe);
                evalWithPunc = true;
                model.A.clear();
                evaluateSet(model, pipe);
            }
			System.out.println();
			System.out.println("_____________________________________________");
			System.out.println();
		}
    }
    
    
    public static void train(PairInstance[] lstTrain, LowRankModel model,
    		PairPipe pipe) throws IOException, InterruptedException {
    	
        if (batchSize <= 0) batchSize = lstTrain.length; 
    	for (int i = 1; i <= maxNumIters; ++i) {
    		System.out.println("=============================================");
    		System.out.printf(" Epoch: %d%n", i);
    		System.out.println("=============================================");
    		
    		long start = System.currentTimeMillis();
    		boolean more = trainIter(i, lstTrain, model, pipe);
    		long end = System.currentTimeMillis();
    		
    		System.out.printf("%nEpoch took %d ms.%n", end-start);    		
    		System.out.println("=============================================");
    		
        	// evaluate on a development set
    		if (test && (i % 10 == 0 || i == maxNumIters)) {		
    			System.out.println();
	  			System.out.println("_____________________________________________");
	  			System.out.println();
	  			System.out.printf(" Evaluation: %s%n", testFile);
	  			System.out.println(); 
	  			double uas = evaluateSet(model, pipe);
	  			System.out.println();
	  			System.out.println("_____________________________________________");
	  			System.out.println();
	  			if (bestDevUAS < uas) {
	  				bestDevUAS = uas;
	  				saveModel();
	  			}
    		}  		
    		
    		System.out.println();
    		System.out.println();
            if (!more) break;
    	}
    	
    }
    
    public static boolean trainIter(int iIter, PairInstance[] lstTrain, 
    		LowRankModel model, PairPipe pipe) throws IOException, InterruptedException {
        
        boolean more = false;
        DecodingThread[] threads = new DecodingThread[numThreads];

        //int[] rndList = new int[lstTrain.length];
        //Random rnd = new Random(System.currentTimeMillis());
        //for (int i = 0, N = lstTrain.length; i < N; ++i) {
        //    int j = rnd.nextInt(i+1);
        //    rndList[i] = rndList[j];
        //    rndList[j] = i;
        //}
        
        double loss = 0.0;
        long timeGrad = 0, timeOpt = 0;
        for (int i = 0, N = lstTrain.length; i < N; i += batchSize) {
            
            //System.out.println();
        	//System.out.printf("  Calculate gradients...");
        	long start = System.currentTimeMillis();
        	taskQueue.clear();
        	for (int j = i; j < N && j < i + batchSize; ++j) {
                DecodingInstance dinst = new DecodingInstance();
                PairInstance inst = lstTrain[j];
                dinst.index = j;
                dinst.inst = inst;
                int n = dinst.inst.length;
                //dinst.arcScores = new double[inst.length][inst.length];
    		    //dinst.ntScores = new double[inst.length][pipe.types.length][2][2];
    		    //model.calculateScores(dinst.inst, dinst.arcScores, dinst.ntScores);
                taskQueue.add(dinst);
                //taskQueue.add(j); 
            }
        	
        	int K = taskQueue.size();

          // A: test
          //System.out.println(K);
        	
        	for (int j = 0; j < numThreads; ++j)
        		threads[j] = new DecodingThread(taskQueue, model);
        	for (int j = 0; j < numThreads; ++j) {
        		threads[j].join();
        		more |= threads[j].more;
                loss += threads[j].loss;
        	}
        	long end = System.currentTimeMillis();
            timeGrad += (end - start);
        	//System.out.printf("%d ms.%n", end-start);
            
            start = System.currentTimeMillis();
            model.optimize(K);
            end = System.currentTimeMillis();
            timeOpt += (end - start);
        	
//        	// evaluate on a development set
//            if (test && (model.iters % 1000 == 0 || model.iters == maxNumIters)) {		
//    			System.out.println();
//    			System.out.println("_____________________________________________");
//    			System.out.println();
//    			System.out.printf(" Evaluation: %s%n", testFile);
//    			System.out.println(); 
//    			double uas = evaluateSet(model, pipe);
//    			System.out.println();
//    			System.out.println("_____________________________________________");
//    			System.out.println();
//    			if (bestDevUAS < uas) {
//    				bestDevUAS = uas;
//    				saveModel();
//    			}
//    		}  		

        }
    	
        System.out.printf(" Gradient calculation took %d ms.%n", timeGrad);
        System.out.printf(" Optimization took %d ms.%n", timeOpt);
        System.out.printf(" Total loss = %f%n", loss / lstTrain.length);
        return more;
    }
    
    public static double evaluateSet(LowRankModel model, PairPipe pipe, boolean output)
    		throws IOException, InterruptedException {
    	
    	BufferedReader in = new BufferedReader(
    			new InputStreamReader(new FileInputStream(testFile),"UTF8"));
    	
    	BufferedWriter out = null;
    	if (output && outFile != null) {
    		out = new BufferedWriter(
    			new OutputStreamWriter(new FileOutputStream(outFile), "UTF8"));
    	}
    	
    	DepDecoder decoder = new DepDecoder();    	
    	int nUCorrect = 0, nLCorrect = 0;
    	int nDeps = 0, nWhole = 0, nSents = 0;
    	
    	PairInstance inst = pipe.createInstance(in);
    	while (inst != null) {
    		++nSents;
            
            int nToks = 0;
            if (evalWithPunc)
    		    nToks = (inst.length - 1);
            else {
                for (int i = 1; i < inst.length; ++i) {
                	//if (inst.sentence[i].matches("[-!\"#%&'()*,./:;?@\\[\\]_{}、]+")) continue;
                    //if ( (inst.posC != null && inst.posC[i].equals(".")) ||
                    //     (inst.posC == null && inst.sentence[i].matches("[,.:'`]+"))) continue;
                    ++nToks;
                }
            }
            nDeps += nToks;
    		
    		//double[][] arcScores = new double[inst.length][inst.length];
    		//double[][][][] ntScores = new double[inst.length][pipe.types.length][2][2];
    		//model.calculateScores(inst, arcScores, ntScores);
    		
    		//String bestParse = decoder.decodeProjective(inst, arcScores, ntScores);
    		//String bestParse = decoder.decodeProjective(inst, arcScores, ntScores, false);

        /*
    		int ua = evaluateUnlabelCorrect(inst, inst.actParseTree, bestParse), la = 0;
    		if (PairPipe.learnLabel)
    			la = evaluateLabelCorrect(inst, inst.actParseTree, bestParse);
    		nUCorrect += ua;
    		nLCorrect += la;
    		if ((PairPipe.learnLabel && la == nToks) ||
    				(!PairPipe.learnLabel && ua == nToks)) 
    			++nWhole;
    		
    		if (out != null) {
    			int[] deps = new int[inst.length], labs = new int[inst.length];
    			model.getParseDepsAndLabs(bestParse, deps, labs);
    			String line1 = "", line2 = "", line3 = "", line4 = "";
    			for (int i = 1; i < inst.length; ++i) {
    				line1 += inst.sentence[i] + "\t";
    				line2 += inst.pos[i] + "\t";
    				line3 += (PairPipe.learnLabel ? pipe.types[labs[i]] : labs[i]) + "\t";
    				line4 += deps[i] + "\t";
    			}
    			out.write(line1.trim() + "\n" + line2.trim() + "\n" + line3.trim() + "\n" + line4.trim() + "\n\n");
    		}
        */
    		
    		inst = pipe.createInstance(in);
    	}
    	
    	in.close();
    	if (out != null) out.close();
    	System.out.printf(" Tokens: %d%n", nDeps);
    	System.out.printf(" Sentences: %d%n", nSents);
    	System.out.printf(" UAS=%.6f\tLAS=%.6f\tCAS=%.6f%n",
    			(nUCorrect+0.0)/nDeps,
    			(nLCorrect+0.0)/nDeps,
    			(nWhole + 0.0)/nSents);
    	
    	return (nUCorrect+0.0)/nDeps;
    }
    
    public static double evaluateSet(LowRankModel model, PairPipe pipe) 
    		throws IOException, InterruptedException  {
    	
    	return evaluateSet(model, pipe, false);
    	
    }

    public static int evaluateUnlabelCorrect(DepInstance inst, String act, String pred) {
    	int nCorrect = 0;
    	String[] actParts = act.split(" ");
    	String[] predParts = pred.split(" ");
    	for (int i = 0; i < actParts.length; ++i) {

            if (!evalWithPunc)
            	if (inst.sentence[i+1].matches("[-!\"#%&'()*,./:;?@\\[\\]_{}、]+")) continue;
                //if ( (inst.posC != null && inst.posC[i+1].equals(".")) ||
                //    (inst.posC == null && inst.sentence[i+1].matches("[,.:'`]+"))) continue;

    		String p = actParts[i].split(":")[0];
    		String q = predParts[i].split(":")[0];
    		if (p.equals(q)) ++nCorrect;
    	}    		
    	return nCorrect;
    }
    
    public static int evaluateLabelCorrect(DepInstance inst, String act, String pred) {
    	int nCorrect = 0;
    	String[] actParts = act.split(" ");
    	String[] predParts = pred.split(" ");
    	for (int i = 0; i < actParts.length; ++i) {

            if (!evalWithPunc)
            	if (inst.sentence[i+1].matches("[-!\"#%&'()*,./:;?@\\[\\]_{}、]+")) continue;
                //if ( (inst.posC != null && inst.posC[i+1].equals(".")) ||
                //    (inst.posC == null && inst.sentence[i+1].matches("[,.:'`]+"))) continue;

    		if (actParts[i].equals(predParts[i])) ++nCorrect;
    	}    		
    	return nCorrect;
    }
    

    public static int evaluateUnlabelCorrect(String act, String pred) {
    	int nCorrect = 0;
    	String[] actParts = act.split(" ");
    	String[] predParts = pred.split(" ");
    	for (int i = 0; i < actParts.length; ++i) {
    		String p = actParts[i].split(":")[0];
    		String q = predParts[i].split(":")[0];
    		if (p.equals(q)) ++nCorrect;
    	}    		
    	return nCorrect;
    }
    
    public static int evaluateLabelCorrect(String act, String pred) {
    	int nCorrect = 0;
    	String[] actParts = act.split(" ");
    	String[] predParts = pred.split(" ");
    	for (int i = 0; i < actParts.length; ++i) {
    		if (actParts[i].equals(predParts[i])) ++nCorrect;
    	}    		
    	return nCorrect;
    }
    
    private static void processArguments(String[] args) {
    	
    	for (String arg : args) {
    		if (arg.equals("train")) {
    			train = true;
    		}
    		else if (arg.equals("test")) {
    			test = true;
    		}
    		else if (arg.equals("label")) {
    			DepPipe.learnLabel = true;
    		}
    		else if (arg.startsWith("train-file:")) {
    			trainFile = arg.split(":")[1];
    		}
    		else if (arg.startsWith("test-file:")) {
    			testFile = arg.split(":")[1];
    		}
    		else if (arg.startsWith("output-file:")) {
    			outFile = arg.split(":")[1];
    		}
    		else if (arg.startsWith("model-file:")) {
    			modelFile = arg.split(":")[1];
    		}
            else if (arg.startsWith("max-sent:")) {
                DepPipe.maxNumSent = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("lambda:")) {
                LowRankModel.lambdaEta = Double.parseDouble(arg.split(":")[1]);
            }
            else if (arg.startsWith("cc:")) {
                LowRankModel.cc = Double.parseDouble(arg.split(":")[1]);
            }
            else if (arg.startsWith("C:")) {
            	LowRankModel.C = Double.parseDouble(arg.split(":")[1]);
            }
            else if (arg.startsWith("R:")) {
                LowRankMatrix.R = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("threads:")) {
            	numThreads = Integer.parseInt(arg.split(":")[1]);
                LowRankModel.numThreads = numThreads;
            }
            else if (arg.startsWith("word-vector:")) {
            	wordVectorFile = arg.split(":")[1];
            }
            else if (arg.startsWith("loss:")) {
                String type = arg.split(":")[1];
                if (type.equals("hinge"))
                    LowRankModel.lossType = LowRankModel.LossType.HINGE;
                else if (type.equals("smooth"))
                    LowRankModel.lossType = LowRankModel.LossType.SMOOTHED_HINGE;
            }
            else if (arg.startsWith("iters:")) {
                maxNumIters = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("batch:")) {
            	batchSize = Integer.parseInt(arg.split(":")[1]);
            }
    	}
    	
    	System.out.println("------\nFLAGS\n------");
    	System.out.println("train-file: " + trainFile);
    	System.out.println("test-file: " + testFile);
    	System.out.println("model-name: " + modelFile);
        System.out.println("output-file: " + outFile);
    	System.out.println("train: " + train);
    	System.out.println("test: " + test);
        System.out.println("iters: " + maxNumIters);
    	System.out.println("label: " + DepPipe.learnLabel);
        System.out.println("max-sent: " + DepPipe.maxNumSent);      
        System.out.println("lambdaEta: " + LowRankModel.lambdaEta);
        System.out.println("C: " + LowRankModel.C);
        System.out.println("cc: " + LowRankModel.cc);
        System.out.println("R: " + LowRankMatrix.R);
        System.out.println("NumThreads: " + numThreads);
        System.out.println("word-vector:" + wordVectorFile);
        System.out.println("loss: " + LowRankModel.lossType);
    	System.out.println("------\n");
    }
    
    public static void saveModel() throws IOException {
    	ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(modelFile));
    	//out.writeObject(pipe.wordAlphabet);
    	//out.writeObject(pipe.arcAlphabet);
    	//out.writeObject(pipe.typeAlphabet);
      //  out.writeObject(pipe.seenBigramAlphabet);
    	out.writeObject(model);
    	out.close();
    }
    
    public static void loadModel() throws Exception {
    	ObjectInputStream in = new ObjectInputStream(new FileInputStream(modelFile));
    	pipe = new PairPipe();
    	//pipe.wordAlphabet = (Alphabet) in.readObject();
    	//pipe.arcAlphabet = (Alphabet) in.readObject();
    	//pipe.typeAlphabet = (Alphabet) in.readObject();
        Object obj = in.readObject();
        //if (obj instanceof Alphabet) {
        //    pipe.seenBigramAlphabet = (Alphabet) obj;
        //    obj = in.readObject();
        //}
    	model = (LowRankModel) obj;
    	in.close();
    	//pipe.closeAlphabets();
      //  pipe.seenBigramAlphabet.stopGrowth();
    }
}

class DecodingInstance {
    public int index;
    //public DepInstance inst;
    public PairInstance inst;
    public double[][] arcScores;
    public double[][][][] ntScores;
    
    public DecodingInstance() {}
}

class DecodingThread extends Thread {
	
	public ConcurrentLinkedQueue<DecodingInstance> taskQueue;
    public LowRankModel model;
	public DepDecoder decoder;
	public boolean more;
    public double loss;

	public DecodingThread(ConcurrentLinkedQueue<DecodingInstance> taskQueue,
                            LowRankModel model) {	
		this.taskQueue = taskQueue;
        this.model = model;
		this.decoder = new DepDecoder();
		this.more = false;
        loss = 0.0;		
		start();
	}
	
	public void run() {
		for (DecodingInstance dinst = taskQueue.poll(); dinst != null; dinst = taskQueue.poll()) {
			//DepInstance inst = dinst.inst;
			PairInstance inst = dinst.inst;
      
      /*
      String bestParse = decoder.decodeProjective(inst, dinst.arcScores, 
                                            dinst.ntScores, true);
      int ua = LowRankParser.evaluateUnlabelCorrect(inst.actParseTree, bestParse), la = 0;
    		if (DepPipe.learnLabel)
    			la = LowRankParser.evaluateLabelCorrect(inst.actParseTree, bestParse);
  */
    	//	if ((DepPipe.learnLabel && la != inst.length-1) ||
    	//			(!DepPipe.learnLabel && ua != inst.length-1)) {
    			//double l = model.addConstraint(inst, bestParse);
          String bestParse = "";
    			double l = model.addConstraint(inst, bestParse);
                loss += l;
                more |= (l > 0);
      //      }
		}
		
	}
}
