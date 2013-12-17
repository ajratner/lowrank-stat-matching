package utils;

public class MatrixEntry {
	
	public int x, y;
	public double value;
	public MatrixEntry next;
	
	public MatrixEntry(int _id, double _value, MatrixEntry _next) {
		x = _id;
		y = 1;
		value = _value;
		next = _next;
	}
	
	public MatrixEntry(int _x, int _y, double _value, MatrixEntry _next) {
		x = _x;
		y = _y;
		value = _value;
		next = _next;
	}
	

}
