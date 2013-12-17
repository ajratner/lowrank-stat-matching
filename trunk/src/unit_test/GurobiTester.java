package unit_test;

import gurobi.*;

public class GurobiTester {

	/**
	 * @param args
	 * @throws GRBException 
	 */
	public static void main(String[] args) throws GRBException {
		// TODO Auto-generated method stub
		
		RunTest1();
	}
	
	public static void RunTest1() throws GRBException {
		GRBEnv grbEnv = new GRBEnv("qb.log");
        GRBModel grbModel = new GRBModel(grbEnv);
        GRBQuadExpr grbObj = new GRBQuadExpr();
        
        GRBVar x = grbModel.addVar(0, 1, 0.0, GRB.CONTINUOUS, "x");
        GRBVar y = grbModel.addVar(0, 1, 0, GRB.CONTINUOUS, "y");
        grbModel.update();
        
        grbObj.addTerm(-2, x);
        grbObj.addTerm(-1, y);
        grbObj.addTerm(1, x, x);
        grbObj.addTerm(-2, x, y);
        grbObj.addTerm(1, y, y);
        grbModel.setObjective(grbObj);
        
        GRBLinExpr c1 = new GRBLinExpr();
        c1.addTerm(1.0, x);
        grbModel.addConstr(c1, GRB.LESS_EQUAL, 0.5, "c1");
        GRBLinExpr c2 = new GRBLinExpr();
        c2.addTerm(1.0, y);
        grbModel.addConstr(c2, GRB.LESS_EQUAL, 0.5, "c2");
        
        grbModel.optimize();
        
    	GRBConstr[] constrs = grbModel.getConstrs();
    	double[] PIs = grbModel.get(GRB.DoubleAttr.Pi, constrs);
    	String[] Ids = grbModel.get(GRB.StringAttr.ConstrName, constrs);
    	for (int i = 0; i < constrs.length; ++i)
    		System.out.println(Ids[i] + " " + PIs[i]);

        // now remove a variable
        grbModel.remove(y);
        grbModel.optimize();

	}
}
