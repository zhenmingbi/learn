/**
 * Created by zhenming on 10/28/17.
 */
public class Measures {
    public static double accuracy(int[] y, int[] y_pre){
        double count=0;
        int l=y.length;

        for(int i=0; i<l;i++){
            if(y[i]==y_pre[i]){
                count+=1;
            }
        }
        return count/l;
    }

    public static double RMSE(double[] y, double[] y_predict){
        int l=y.length;
        double s=0.0;
        for(int i=0;i<l;i++){
            s+=(y[i]-y_predict[i])*(y[i]-y_predict[i]);

        }
        return Math.sqrt(s/l);

    }
}
