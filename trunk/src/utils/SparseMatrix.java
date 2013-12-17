package utils;

import java.util.HashMap;


public class SparseMatrix {
	
	int nRows = 1;
	int nCols = 1;
	int size = 0;
	//MatrixEntry element = null;
	
	int capacity;
	int[] x, y;
	double[] z;
		
	public SparseMatrix() { grow(); };
	
	public SparseMatrix(int _nRows) {		
		nRows = _nRows;
		nCols = 1;
		grow();
	}
	
	public SparseMatrix(int _nRows, int _nCols) {
		nRows = _nRows;
		nCols = _nCols;
		grow();
	}
	
	private void grow() {
		
		int cap = 5 > capacity ? 10 : capacity * 2;
		
		int[] x2 = new int[cap], y2 = new int[cap];
		double[] z2 = new double[cap];
		
		if (capacity > 0) {
			System.arraycopy(x, 0, x2, 0, capacity);
			System.arraycopy(y, 0, y2, 0, capacity);
			System.arraycopy(z, 0, z2, 0, capacity);
		}
		
		x = x2;
		y = y2;
		z = z2;
		capacity = cap;
	}
	
	public void addEntry(int _x, double _value) {
		if (_value == 0) return;
		
		if (size == capacity) grow();
		x[size] = _x;
		y[size] = 0;
		z[size] = _value;
		++size;
	}
	
	public void addEntry(int _x, int _y, double _value) {
		if (_value == 0) return;
				
		if (size == capacity) grow();
		x[size] = _x;
		y[size] = _y;
		z[size] = _value;
		++size;
	}
	
	//public void addEntry(int _x, int _y, float _value) {
	//	if (_value == 0) return;
	//			
	//	if (size == capacity) grow();
	//	x[size] = _x;
	//	y[size] = _y;
	//	z[size] = _value;
	//	++size;
	//}

	public void addEntries(SparseMatrix m) {
		addEntries(m, 1.0);
	}
	
	public void addEntries(SparseMatrix m, double coeff) {
		
		assert(m != null && m.nRows == nRows && m.nCols == nCols);
		if (coeff == 0 || m.size == 0) return;
		
		for (int i = 0; i < m.size; ++i)
			addEntry(m.x[i], m.y[i], m.z[i] * coeff);
	}
	
	public boolean isVector() {
		return nCols == 1;
	}
	
	public void rescale(double coeff) {
		for (int i = 0; i < size; ++i)
			z[i] *= coeff;
	}
	
	public double l2Norm() {
		double sum = 0;
		for (int i = 0; i < size; ++i)
			sum += z[i]*z[i];
		return Math.sqrt(sum);
	}
	
    public double min() {
        double m = Double.POSITIVE_INFINITY;
        for (int i = 0; i < size; ++i)
            if (m > z[i]) m = z[i];
        return m;
    }
    
    public double max() {
        double m = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < size; ++i)
            if (m < z[i]) m = z[i];
        return m;
    }

	public int size() {
		return size;
	}
    
    public int nRows() { return nRows; }
    public int nCols() { return nCols; }

	public int[] x() {
		int[] xx = new int[size];
		if (size > 0)
			System.arraycopy(x, 0, xx, 0, size);
		return xx;
	}
	
	public int[] y() {
		int[] yy = new int[size];
		if (size > 0)
			System.arraycopy(y, 0, yy, 0, size);
		return yy;
	}
	
	public double[] z() {
		double[] zz = new double[size];
		if (size > 0)
			System.arraycopy(z, 0, zz, 0, size);
		return zz;
	}

    public void setZ(int i, double v) { z[i] = v; }
	public int x(int i) { return x[i]; }
	public int y(int i) { return y[i]; }
	public double z(int i) { return z[i]; }
	
	public boolean aggregate() {
		
		if (size == 0) return false;
		
		boolean aggregated = false;
		
		HashMap<Long, MatrixEntry> table = new HashMap<Long, MatrixEntry>();
		long N = nRows, M = nCols;
		for (int i = 0; i < size; ++i) {
			Long id = M * x[i] + y[i];
			MatrixEntry item = table.get(id);
			if (item != null) {
				item.value += z[i];
				aggregated = true;
			} else
				table.put(id, new MatrixEntry(x[i], y[i], z[i], null));
		}
		
		if (!aggregated) return false;
		
		int p = 0;
		for (MatrixEntry e: table.values()) 
			if (e.value != 0) {
				x[p] = e.x;
				y[p] = e.y;
				z[p] = e.value;
				++p;
			}
		size = p;
		return true;
	}
	
    public double dotProduct(SparseMatrix _y) {
        return dotProduct(this, _y);
    }
        
	public double dotProduct(double[] _y) {
		return dotProduct(this, _y);
	}
	
	
	public double leftRightMultiply(double[] u, double[] v) {
		
		assert(nRows == u.length && nCols == v.length);
		
		double s = 0;
		for (int i = 0; i < size; ++i)
			s += u[x[i]] * v[y[i]] * z[i];
		return s;
	}
	
	private static double[] dpVec;			 //non-sparse vector repr for vector dot product
	public static double dotProduct(SparseMatrix _x, SparseMatrix _y) {
		
		assert(_x.isVector() && _y.isVector() && _x.nRows == _y.nRows);		
		
		if (dpVec == null || dpVec.length < _y.nRows) dpVec = new double[_y.nRows];
		
		for (int i = 0; i < _y.size; ++i)
			dpVec[_y.x[i]] += _y.z[i];
		
		double sum = 0;
		for (int i = 0; i < _x.size; ++i)
			sum += _x.z[i] * dpVec[_x.x[i]];

		for (int i = 0; i < _y.size; ++i)
			dpVec[_y.x[i]] = 0;
		
		return sum;
	}
	
	public static double dotProduct(SparseMatrix _x, double[] _y) {
		
		assert(_x.isVector() && _x.nRows == _y.length);	
		
		double sum = 0;
		for (int i = 0; i < _x.size; ++i)
			sum += _x.z[i] * _y[_x.x[i]];
		return sum;
	}
	
	public static SparseMatrix outerProduct(SparseMatrix _x, SparseMatrix _y) {
		
		assert(_x.isVector() && _y.isVector());
		
		SparseMatrix mat = new SparseMatrix(_x.nRows, _y.nRows);
		for (int i = 0; i < _x.size; ++i)
			for (int j = 0; j < _y.size; ++j)
				mat.addEntry(_x.x[i], _y.x[j], _x.z[i] * _y.z[j]);

		return mat;
	}
	
}


