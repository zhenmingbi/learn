import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhenming on 11/1/17.
 */
public class Treemodel{
    public static List<Integer> featureindex(String s)throws Exception {
        List<Integer> featureindexoftree=new ArrayList<>();
        GradientBoostingClassification model = (GradientBoostingClassification) Serialization.deserialize(s);
        for (List<RegDataSetRegressionTree> treeofClasses : model.trees){
            for(RegDataSetRegressionTree oldtree : treeofClasses){
                featureindexoftree.add(oldtree.root.index);
            }
       }
        return featureindexoftree;
    }

    public static List<Integer> featureindexnew(String s)throws Exception {
        List<Integer> featureindexoftree=new ArrayList<>();
        FeatureBoostingClassification model = (FeatureBoostingClassification) Serialization.deserialize(s);
        for (List<FeatureRegressionTree> treeofClasses : model.trees){
            for(FeatureRegressionTree oldtree : treeofClasses){
                featureindexoftree.add(oldtree.root.index);
            }
        }
        return featureindexoftree;
    }

    public static void main(String[] args)throws Exception {
        List<Integer> featureIndexOld =featureindex("/Users/zhenming/Learn/Boosting/oldBoostingI800/optimized/model");
        File oldtreeindexFile = new File("/Users/zhenming/Learn/Boosting/oldBoostingI800/optimized","bestIndexofTree");
        FileUtils.writeStringToFile(oldtreeindexFile, featureIndexOld.toString());
        List<Integer> featureIndexnew8 =featureindexnew("/Users/zhenming/Learn/Boosting/newBoostingI800/8/optimized/model");
        File newtreeindexFile = new File("/Users/zhenming/Learn/Boosting/newBoostingI800/8/optimized","bestIndexofTree");
        FileUtils.writeStringToFile(newtreeindexFile, featureIndexnew8.toString());


    }

}

