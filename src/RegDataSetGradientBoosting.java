import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.StandardFormat;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by zhenming on 10/22/17.
 */
public class RegDataSetGradientBoosting implements Serializable {
    private static final long serialVersionUID = 1L;
    List<RegDataSetRegressionTree> trees=new ArrayList<>();
    double prior;
    transient RegDataSet dataSet;
    transient double[] f;

    public RegDataSetGradientBoosting(RegDataSet dataSet) {
        this.dataSet = dataSet;
        double[] y = dataSet.getLabels();
        int l=y.length;
        prior=MyMath.avg(y);
        f=new double[l];
        Arrays.fill(f, prior);
    }

    public double predict(Vector x){
        double s=prior;
        for (int i=0; i<trees.size();i++){
            s+=0.1*trees.get(i).single_predict(x);
        }
        return s;

    }

    public double[] predict(RegDataSet x){
        int l=x.getNumDataPoints();
        double[] res=new double[l];
        for (int i=0; i<l;i++){
            res[i]=predict(x.getRow(i));
        }
        return res;

    }




    public void iterate(){
        double[] y = dataSet.getLabels();
        int l = y.length;
        double[] r=new double[l];
        for (int i=0; i<dataSet.getNumDataPoints(); i++){
            r[i]=y[i]-f[i];
        }
        RegDataSetRegressionTree regressor=new RegDataSetRegressionTree();
        regressor.fit(dataSet, r,2);
        trees.add(regressor);
        for (int j=0; j<l; j++){
            f[j]+=0.1*regressor.single_predict(dataSet.getRow(j));
        }
    }




    public static void main(String[] args) throws Exception {
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }
        Config config = new Config(args[0]);
        testRMSE(config);
        trainingTime(config);

    }

    private static void testRMSE(Config config)throws Exception{
        RegDataSet x_train= TRECFormat.loadRegDataSet(config.getString("input.trainSet"), DataSetType.REG_SPARSE, true);
        RegDataSet x_test=TRECFormat.loadRegDataSet(config.getString("input.testSet"),DataSetType.REG_SPARSE, true);
        double[] y_test=x_test.getLabels();
        List<Double> rmse=new ArrayList<>();
        RegDataSetGradientBoosting regressor = new RegDataSetGradientBoosting(x_train);
        for (int iter=1;iter<=config.getInt("iterations");iter++){
            System.out.println("iteration="+iter);
            regressor.iterate();
            double[] y_predict = regressor.predict(x_test);
            rmse.add(RegressionTree.RMSE(y_test, y_predict));
        }
        System.out.println(rmse);
        File outputDir = new File(config.getString("output"));
        outputDir.mkdirs();
        File model=new File(outputDir, "model");
        Serialization.serialize(regressor,model);
        File rmseFile = new File(outputDir,"rmse");
        FileUtils.writeStringToFile(rmseFile, rmse.toString());
        config.store(new File(outputDir,"config"));

    }

    private static void trainingTime(Config config) throws Exception{
        RegDataSet x_train= TRECFormat.loadRegDataSet(config.getString("input.trainSet"), DataSetType.REG_SPARSE, true);
        RegDataSet x_test=TRECFormat.loadRegDataSet(config.getString("input.testSet"),DataSetType.REG_SPARSE, true);
        double[] y_test=x_test.getLabels();
        List<Double> time=new ArrayList<>();
        RegDataSetGradientBoosting regressor = new RegDataSetGradientBoosting(x_train);
        StopWatch stopWatch=new StopWatch();
        stopWatch.start();
        for (int iter=1;iter<=config.getInt("iterations");iter++){
            System.out.println("iteration="+iter);
            regressor.iterate();
            time.add(stopWatch.getTime()/1000.0);
        }
        System.out.println(time);
        File outputDir = new File(config.getString("output"));
        outputDir.mkdirs();
        File timeFile = new File(outputDir,"trainingTime");
        FileUtils.writeStringToFile(timeFile, time.toString());


    }
}
