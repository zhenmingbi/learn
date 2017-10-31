import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.util.MathUtil;
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
 * Created by zhenming on 10/28/17.
 */
public class GradientBoostingClassification implements Serializable {
        private static final long serialVersionUID = 1L;
        List<List<RegDataSetRegressionTree>> trees=new ArrayList<>();
        double[] prior;
        transient ClfDataSet dataSet;
        transient double[][] f;


        public GradientBoostingClassification(ClfDataSet dataSet) {
            this.dataSet = dataSet;
            int[] y = dataSet.getLabels();
            int l=y.length;
            int m=dataSet.getNumClasses();
            prior = MathUtil.inverseSoftMax(initalprobs(y));
            f=new double[l][m];
            for(int i=0;i<l;i++){
                for(int j=0;j<m;j++){
                    f[i][j]=prior[j];
                }
            }
            for(int i=0; i<m;i++){
                trees.add(new ArrayList<>());
            }
        }

        public double[] initalprobs(int[] y){
            int m=dataSet.getNumClasses();
            int l=y.length;
            double[] count=new double[m];
            double [] p=new double[m];

            for(int i=0; i<l; i++){
                count[y[i]]+=1;
            }
            for(int j=0;j<m;j++){
                p[j]=count[j]/l;
            }
            return p;

        }

        public double[] scoreToprobs(double[]s){
            int l=s.length;
            double[] p=new double[l];
            double lse=MathUtil.logSumExp(s);
            for(int i=0;i<l;i++){
                p[i]=Math.exp(s[i]-lse);

            }
            return p;
        }

        public int predict(Vector x){
            int m=dataSet.getNumClasses();
            double[] s= Arrays.copyOf(prior,m);
            for (int i=0; i<m;i++){
                for(RegDataSetRegressionTree tree: trees.get(i)){
                    s[i]+=0.1*tree.single_predict(x);
                }
            }
            return MyMath.maxIndex(s);

        }



        public int[] predict(ClfDataSet x){
            int l=x.getNumDataPoints();
            int[] res=new int[l];
            for (int i=0; i<l;i++){
                res[i]=predict(x.getRow(i));
            }
            return res;

        }


        public void iterate(){
            int[] y = dataSet.getLabels();
            int l = y.length;
            int m=dataSet.getNumClasses();
            double[][] r=new double[m][l];
            for(int i=0; i<l;i++){
                double[] p=scoreToprobs(f[i]);
                for(int j=0;j<m;j++){
                    if(j==y[i]){
                        r[j][i]=1-p[j];
                    }else{
                        r[j][i]=-p[j];
                    }

                }

            }

            for(int i=0;i<m;i++){
                RegDataSetRegressionTree regressor=new RegDataSetRegressionTree();
                regressor.fit(dataSet, r[i],2);
                trees.get(i).add(regressor);
                for(int j=0;j<l;j++){
                    f[j][i]+=0.1*regressor.single_predict(dataSet.getRow(j));
                }

            }
        }


        public static void main(String[] args) throws Exception {
            if (args.length !=1){
                throw new IllegalArgumentException("Please specify a properties file.");
            }

            Config config = new Config(args[0]);

            testAccuracy(config);
            trainingTime(config);
        }

        private static void testAccuracy(Config config) throws Exception{
            ClfDataSet x_train= TRECFormat.loadClfDataSet(config.getString("input.trainSet"), DataSetType.CLF_SPARSE, true);
            GradientBoostingClassification regressor = new GradientBoostingClassification(x_train);
            ClfDataSet x_test=TRECFormat.loadClfDataSet(config.getString("input.testSet"),DataSetType.CLF_SPARSE, true);
            int[] y_test=x_test.getLabels();


            List<Double> accuracy=new ArrayList<>();
            for (int iter=1;iter<=config.getInt("iterations");iter++){
                System.out.println("iteration="+ iter);
                regressor.iterate();
                int[] y_predict = regressor.predict(x_test);
                accuracy.add(Measures.accuracy(y_test, y_predict));

            }

            System.out.println(accuracy);

            File outputDir = new File(config.getString("output"));
            outputDir.mkdirs();
            File model=new File(outputDir, "model");
            Serialization.serialize(regressor,model);
            File accuracyFile = new File(outputDir,"accuracy");
            FileUtils.writeStringToFile(accuracyFile, accuracy.toString());
            config.store(new File(outputDir,"config"));
        }

        private static void trainingTime(Config config) throws Exception{
            ClfDataSet x_train= TRECFormat.loadClfDataSet(config.getString("input.trainSet"), DataSetType.CLF_SPARSE, true);
            GradientBoostingClassification regressor = new GradientBoostingClassification(x_train);
            List<Double> time=new ArrayList<>();
            StopWatch stopWatch=new StopWatch();
            stopWatch.start();
            for (int iter=1;iter<=config.getInt("iterations");iter++){
                System.out.println("iteration="+ iter);
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

