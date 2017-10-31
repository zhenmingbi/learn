import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by zhenming on 10/20/17.
 */
public class GradientBoosting {
    List<RegressionTree> trees=new ArrayList<>();
    double prior;


    public double predict(double[] x){
        double s=prior;
        for (int i=0; i<trees.size();i++){
            s+=0.1*trees.get(i).single_predict(x);
        }
        return s;

    }

    public double[] predict(double[][]x){
        int l=x.length;
        double[] res=new double[l];
        for (int i=0; i<l;i++){
            res[i]=predict(x[i]);
        }
        return res;

    }



    public void fit(double[][] x, double[] y){
        int l=y.length;
        prior=MyMath.avg(y);
        double[] f=new double[l];
        Arrays.fill(f, prior);
        for(int m=0; m<100;m++){
            double[] r=new double[l];
            for (int i=0; i<x.length; i++){
                r[i]=y[i]-f[i];
            }
            RegressionTree regressor=new RegressionTree();
            regressor.fit(x, r);
            trees.add(regressor);
            for (int j=0; j<l; j++){
                f[j]+=0.1*regressor.single_predict(x[j]);
            }
        }

    }







    public static void main(String[] args)throws Exception {
        double[][] x_train=RegressionTree.load_x("/users/zhenming/Learn/xtrain.csv");
        double[][] x_test=RegressionTree.load_x("/users/zhenming/Learn/xtest.csv");
        double[] y_train=RegressionTree.load_y("/users/zhenming/Learn/ytrain.csv");
        double[] y_test=RegressionTree.load_y("/users/zhenming/Learn/ytest.csv");
        GradientBoosting regressor=new GradientBoosting();
        regressor.fit(x_train,y_train);
        double[] y_predict = regressor.predict(x_test);
        System.out.println(RegressionTree.RMSE(y_test, y_predict));

    }
}
