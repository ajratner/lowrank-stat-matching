package lowrankparser;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Random;


import utils.LowRankMatrix;
import utils.MatrixEntry;
import utils.SVD;
import utils.Rank1Component;
import utils.SparseMatrix;


public class LowRankModel implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public enum LossType {
		HINGE, SMOOTHED_HINGE
	}
    
  //public static boolean lineSearch = true;
  public static int numThreads = 4;
  public static int batchSize = 10;
	public static LossType lossType = LossType.SMOOTHED_HINGE;
	public static double lambdaEta = 0.9;
  public static double cc = 0.001;
  public static double C;

	public transient String[] types;
	public transient Alphabet typeAlphabet;
	
	public int sizeA, sizeEta;	
	public LowRankMatrix A;
	public double[] eta;
	public double etaScale;
	public transient SparseMatrix dA, deta;
	public transient int iters, etaUpdCnt;
  public transient GradientInstance[] gradInsts;
  public transient int numGradInsts = 0;

	//public LowRankModel(DepPipe pipe, int N) throws Exception {
	public LowRankModel(PairPipe pipe, int N) throws Exception {
		
    A = new LowRankMatrix();
    //sizeA = pipe. 

		//eta = new double[pipe.arcAlphabet.size()];
		//etaScale = 1.0;
    //    etaUpdCnt = 0;
		
		//sizeA = pipe.wordAlphabet.size();
	  //sizeEta = pipe.arcAlphabet.size();
		//types = pipe.types;
		//typeAlphabet = pipe.typeAlphabet;
		
		dA = new SparseMatrix(sizeA, sizeA);
		deta = new SparseMatrix(sizeEta);
		iters = 0;
    if (cc != -1) C = (sizeA + sizeEta / lambdaEta) * cc * 0.5;

    gradInsts = new GradientInstance[batchSize];
    numGradInsts = 0;
	}
	
    public void calculateScores(DepInstance inst, double[][] arcScores,
        double[][][][] ntScores) throws InterruptedException {
        
        CalcThread[] threads = new CalcThread[numThreads];
        for (int i = 0; i < numThreads; ++i)
            threads[i] = new CalcThread(numThreads, i, 
                            inst, this, arcScores, ntScores);

        for (int i = 0; i < numThreads; ++i)
            threads[i].join();
    }
    
    public double calculateScore(DepInstance inst, int head, int child) {
      
      double sum = 0;
      if (!DepPipe.constructBigram)
        sum = inst.arcFvs[head][child].dotProduct(eta) * etaScale
          + A.getScore(inst.wordFvs[head], inst.wordFvs[child]);
      else
        sum = inst.arcFvs[head][child].dotProduct(eta) * etaScale
        + A.dotProduct(inst.bigramFvs[head][child]);
      
      return sum;
    }

    // A: simple pair-matching version of addConstraint
    public double addConstraint(PairInstance inst, String parse) {

      int N = inst.length;
      //int[] actDeps = new int[N];
      //int[] actLabs = new int[N];
      //int[] predDeps = new int[N];
      //int[] predLabs = new int[N];
      //getParseDepsAndLabs(inst.actParseTree, actDeps, actLabs);
      //getParseDepsAndLabs(parse, predDeps, predLabs);
      
      //double Fi = getHammingDis(actDeps, actLabs, predDeps, predLabs);
      
      /*
      SparseMatrix ei = new SparseMatrix(sizeEta);
      for (int child = 1; child < N; ++child) {
        int head = actDeps[child];
        int type = actLabs[child];
        int toR = head < child ? 1 : 0;
        ei.addEntries(inst.arcFvs[head][child]);
        if (DepPipe.learnLabel) {
          ei.addEntries(inst.ntFvs[head][type][toR][0]);
          ei.addEntries(inst.ntFvs[child][type][toR][1]);
        }
      }
      for (int child = 1; child < N; ++child) {
        int head = predDeps[child];
        int type = predLabs[child];
        int toR = head < child ? 1 : 0;
        ei.addEntries(inst.arcFvs[head][child], -1.0);
        if (DepPipe.learnLabel) {
          ei.addEntries(inst.ntFvs[head][type][toR][0], -1.0);
          ei.addEntries(inst.ntFvs[child][type][toR][1], -1.0);
        }
      }
      //ei.aggregate();
      */
      
      SparseMatrix hi = new SparseMatrix(sizeA, sizeA);
      for (int child = 1; child < N; ++child) {
        int head = actDeps[child];
        if (!DepPipe.constructBigram) {
          SparseMatrix m = SparseMatrix.outerProduct(
                  inst.wordFvs[head], inst.wordFvs[child]);
          hi.addEntries(m);
        } else {
          SparseMatrix m = inst.bigramFvs[head][child];
          hi.addEntries(m);
        }
      }
      for (int child = 1; child < N; ++child) {
        int head = predDeps[child];
        if (!DepPipe.constructBigram) {
          SparseMatrix m = SparseMatrix.outerProduct(
                inst.wordFvs[head], inst.wordFvs[child]);
          hi.addEntries(m, -1.0);
        } else {
          SparseMatrix m = inst.bigramFvs[head][child];
          hi.addEntries(m, -1.0);
        }
      }
      //hi.aggregate();
        
        
        double ai = A.dotProduct(hi) + ei.dotProduct(eta)*etaScale;
      double xi = - ai + Fi;
      double fxi = getLossGradient(xi);  // new loss gradient here...
      if (fxi != 0) {
        synchronized (dA) {
          dA.addEntries(hi, fxi);
        }
        synchronized (deta) {
          deta.addEntries(ei, fxi);
        }
            
            synchronized (gradInsts) {
                gradInsts[numGradInsts++] = 
                    new GradientInstance(hi, ei, ai, xi);
            }
            return xi;
      }	
        return 0.0;
    }

    
    public void optimize(int N)  {
    	++iters;
        //if (iters % 1000 == 0)
    	//    System.out.printf("  Optimization Iter %d%n", iters);
    	
    	int sda = dA.size(), sdeta = deta.size();
    	dA.aggregate();
    	deta.aggregate();
    	if (iters % 1000 == 0)
    		System.out.printf("  (%d->%d, %d->%d) elements in gradients.%n", 
    				sda, dA.size(), sdeta, deta.size());
    	    	
    	// grad(Loss(data)) = 1/N * \sum grad(loss(sample_i))
    	dA.rescale(1.0/N);
    	deta.rescale(1.0/N);
    	
    	double sigma = 0.0;
    	double[] u = new double[sizeA], v = new double[sizeA];
    	if (dA.size() > 0) sigma = SVD.powerMethod(dA.x(), dA.y(), dA.z(), u, v);    	
    	double l2Eta = deta.l2Norm();
    	//System.out.printf("  sigma(A)=%.6f  |eta|=%.6f  lambda_eta=%f%n",
    	//		sigma, l2Eta, lambdaEta);  
        
        boolean updateEta = l2Eta > sigma * lambdaEta;
        if (updateEta) {
            double coeff = C / lambdaEta / l2Eta;
            deta.rescale(coeff);
        }
        
        double alpha = iters > 1 ? 2.0/iters : 1.0;     // step size;
//        if (lineSearch) {
//            if (updateEta) {
//                for (int i = 0; i < numGradInsts; ++i)
//                    gradInsts[i].bi = - gradInsts[i].hi.leftRightMultiply(u,v) * C;
//            } else {
//                for (int i = 0; i < numGradInsts; ++i)
//                    gradInsts[i].bi = - gradInsts[i].ei.dotProduct(deta);
//            }
//            alpha = runLineSearch();
//            //System.out.println(alpha);
//        }
         
    	if (alpha < 1.0) {
    		A.rescale(1.0-alpha);
    		etaScale *= (1.0-alpha);
            if (A.scale() < 1e-4) A.resetScale();
            if (etaScale < 1e-4) {
                for (int p = 0; p < sizeEta; ++p) eta[p] *= etaScale;
                etaScale = 1.0;
            }
    	} else {
    		A.clear();
    		for (int p = 0; p < sizeEta; ++p) eta[p] = 0;
    	}

    	if (updateEta) {				// largest singular value is C*|eta|/lambda_eta
    	    ++etaUpdCnt;
            alpha /= etaScale;
    		for (int i = 0, S = deta.size(); i < S; ++i)
    			eta[deta.x(i)] += deta.z(i) * alpha;
    		
    	} else {						// largest singular value is C*sigma
    		
    		A.add(alpha * C, u, v);
            //System.out.printf("  ||A|| = %f%n", A.traceNorm());
    	}
        
        if (iters % 1000 == 0)
            System.out.printf("  GD iter: %d  Rank(A)=%d   Updated eta %d times%n", 
        	    iters, A.size(), etaUpdCnt);
    	dA = new SparseMatrix(sizeA, sizeA);
    	deta = new SparseMatrix(sizeEta);
        numGradInsts = 0;
    }
    
    public double runLineSearch() {
        double eps = 1e-8;
        double left = 0.0, right = 1.0;
        double mid, grad, f, x;
        for (double alpha = 0.0; alpha < 1.0; alpha += 0.1) {
            f = 0;
            for (int i = 0; i < numGradInsts; ++i) {
                GradientInstance inst = gradInsts[i];
                x = inst.xi + alpha * (inst.ai + inst.bi);
                f += getLoss(x);
            }
            System.out.printf("%f ", f);
        }
        System.out.println();

        while (left + eps < right) {
            mid = (left+right)/2;
            grad = 0; 
            f = 0;
            for (int i = 0; i < numGradInsts; ++i) {
                GradientInstance inst = gradInsts[i];
                x = inst.xi + mid * (inst.ai + inst.bi);
                grad += getLossGradient(x) * (inst.ai + inst.bi);
            }
            if (grad < 0)
                left = mid;
            else
                right = mid;
        }
        return left;
    }

    private double getLoss(double x) {
        if (lossType == LossType.HINGE) {
    		return x > 0 ? x : 0;
    	} else if (lossType == LossType.SMOOTHED_HINGE) {
    		return x <= 0 ? 0 : (x >= 1 ? x-0.5 : 0.5*x*x);
    	}
    	return 0;
    }

    private double getLossGradient(double x) {
        if (lossType == LossType.HINGE) {
    		return x > 0 ? 1 : 0;
    	} else if (lossType == LossType.SMOOTHED_HINGE) {
    		return x <= 0 ? 0 : (x >= 1 ? 1 : x);
    	}
    	return 0;
    }
    */
    /*
    private double getLossGradient(double x) {
    	if (lossType == LossType.HINGE) {
    		return x < 0 ? -1 : 0;
    	} else if (lossType == LossType.SMOOTHED_HINGE) {
    		return x >=0 ? 0 : (x <= -1 ? -1 : x);
    	}
    	return 0;
    }
   */ 
    public double getHammingDis(int[] actDeps, int[] actLabs,
    			int[] predDeps, int[] predLabs) {
    	double dis = 0;
    	for (int i = 1; i < actDeps.length; ++i)
    		if (DepPipe.learnLabel) {
	    		if (actDeps[i] != predDeps[i]) dis += 1;
	    		if (actLabs[i] != predLabs[i]) dis += 1;
    		} else {
    			if (actDeps[i] != predDeps[i]) dis += 1;
    		}
    	return dis;
    }
    
    public void getParseDepsAndLabs(String parse, int[] deps, int[] labs) {
    	for (String p : parse.split(" ")) {
    		String[] ts = p.split("[:|]");
    		int head = Integer.parseInt(ts[0]);
    		int child = Integer.parseInt(ts[1]);
    		int type = Integer.parseInt(ts[2]);
    		deps[child] = head;
    		labs[child] = type;
    	}
    }
}

class CalcThread extends Thread {
    
    int numThreads, index;
    public LowRankModel model;
    public DepInstance inst;
    public double[][] arcScores;
    public double[][][][] ntScores;

	public CalcThread(int numThreads, int index, 
                    DepInstance inst, LowRankModel model,
                    double[][] arcScores, double[][][][] ntScores) {	
        this.numThreads = numThreads;
        this.index = index;
        this.inst = inst;
        this.model = model;
        this.arcScores = arcScores;
        this.ntScores = ntScores;
		start();
	}
	
	public void run() {

	    int N = inst.length;
        int NN = N * N;
        for (int p = index; p < NN; p += numThreads) {
            int i = p / N;
            int j = p % N;
            if (i != j)
    		    arcScores[i][j] = model.calculateScore(inst, i, j);
    	}
    	
    	if (inst.ntFvs != null && N > 0) {
    		SparseMatrix[][][][] ntFvs = inst.ntFvs;
    		int T = inst.ntFvs[0].length;
            int T4 = T*4;
            int NT4 = N * T * 4;
            for (int p = index; p < NT4; ++p) {
                int i = p / T4;
                int t = (p >> 2) % T;
                int j = (p >> 1) & 1;
                int k = p & 1;
                ntScores[i][t][j][k] = ntFvs[i][t][j][k].dotProduct(model.eta) * model.etaScale;
            }
        }		
	}
}

class GradientInstance {
    public SparseMatrix hi, ei;
    public double ai, bi, xi;
    public GradientInstance(SparseMatrix hi, SparseMatrix ei, double ai, double xi) {
        this.hi = hi;
        this.ei = ei;
        this.ai = ai;
        this.xi = xi;
    }
}
    /*
    public double addConstraint(DepInstance inst, String parse) {
      
      int N = inst.length;
      int[] actDeps = new int[N];
      int[] actLabs = new int[N];
      int[] predDeps = new int[N];
      int[] predLabs = new int[N];
      getParseDepsAndLabs(inst.actParseTree, actDeps, actLabs);
      getParseDepsAndLabs(parse, predDeps, predLabs);
      
      double Fi = getHammingDis(actDeps, actLabs, predDeps, predLabs);

      SparseMatrix ei = new SparseMatrix(sizeEta);
      for (int child = 1; child < N; ++child) {
        int head = actDeps[child];
        int type = actLabs[child];
        int toR = head < child ? 1 : 0;
        ei.addEntries(inst.arcFvs[head][child]);
        if (DepPipe.learnLabel) {
          ei.addEntries(inst.ntFvs[head][type][toR][0]);
          ei.addEntries(inst.ntFvs[child][type][toR][1]);
        }
      }
      for (int child = 1; child < N; ++child) {
        int head = predDeps[child];
        int type = predLabs[child];
        int toR = head < child ? 1 : 0;
        ei.addEntries(inst.arcFvs[head][child], -1.0);
        if (DepPipe.learnLabel) {
          ei.addEntries(inst.ntFvs[head][type][toR][0], -1.0);
          ei.addEntries(inst.ntFvs[child][type][toR][1], -1.0);
        }
      }
      //ei.aggregate();
      
      SparseMatrix hi = new SparseMatrix(sizeA, sizeA);
      for (int child = 1; child < N; ++child) {
        int head = actDeps[child];
        if (!DepPipe.constructBigram) {
          SparseMatrix m = SparseMatrix.outerProduct(
                  inst.wordFvs[head], inst.wordFvs[child]);
          hi.addEntries(m);
        } else {
          SparseMatrix m = inst.bigramFvs[head][child];
          hi.addEntries(m);
        }
      }
      for (int child = 1; child < N; ++child) {
        int head = predDeps[child];
        if (!DepPipe.constructBigram) {
          SparseMatrix m = SparseMatrix.outerProduct(
                inst.wordFvs[head], inst.wordFvs[child]);
          hi.addEntries(m, -1.0);
        } else {
          SparseMatrix m = inst.bigramFvs[head][child];
          hi.addEntries(m, -1.0);
        }
      }
      //hi.aggregate();
        
        
        double ai = A.dotProduct(hi) + ei.dotProduct(eta)*etaScale;
      double xi = - ai + Fi;
      double fxi = getLossGradient(xi);
      if (fxi != 0) {
        synchronized (dA) {
          dA.addEntries(hi, fxi);
        }
        synchronized (deta) {
          deta.addEntries(ei, fxi);
        }
            
            synchronized (gradInsts) {
                gradInsts[numGradInsts++] = 
                    new GradientInstance(hi, ei, ai, xi);
            }
            return xi;
      }	
        return 0.0;
    }
    */
