/**
 * Created by zhenming on 10/20/17.
 */
public class MyMath {
    public static double getMax(double[] array) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
            }
        }
        return max;
    }

    public static int maxIndex(double[] array){
        double max=Double.NEGATIVE_INFINITY;
        int mindex=0;
        for(int i=0;i<array.length;i++){
            if(array[i]>max){
                max=array[i];
                mindex=i;
            }
        }
        return mindex;
    }

    public static double getMin(double[] array) {
        double min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < array.length; i++) {
            if (array[i] < min) {
                min = array[i];
            }
        }
        return min;
    }


    public static double sum(double[] array) {
        double s = 0.0;
        for (int i = 0; i < array.length; i++) {
            s += array[i];
        }
        return s;
    }

    public static int bsum(boolean[] b) {
        int s = 0;
        for (int i = 0; i < b.length; i++) {
            if (b[i]) {
                s++;
            }
        }
        return s;
    }

    public static int sum(int[] array) {
        int s = 0;
        for (int i = 0; i < array.length; i++) {
            s += array[i];
        }
        return s;
    }

    public static boolean[] fill(boolean[] a) {
        for (int i = 0; i < a.length; i++) {
            a[i] = true;
        }
        return a;
    }

    public static double avg(double[] y) {
        return sum(y)/y.length;

    }


//    public static void main(String[] args) {
//
//    }




}
