import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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
        Set<Integer> sOld=new HashSet<>();
        Set<Integer> sNew=new HashSet<>();
        String s="";
        GradientBoostingClassification model = (GradientBoostingClassification) Serialization.deserialize("/Users/zhenming/Learn/Boosting/oldBoostingI800/optimized/model");
        FeatureBoostingClassification modelnew = (FeatureBoostingClassification) Serialization.deserialize("/Users/zhenming/Learn/Boosting/newBoostingI800/8/optimized/model");
        int cY=0;
        int cN=0;
        for(int i=0; i<800;i++){
            if(model.trees.get(1).get(i).root.index == modelnew.trees.get(1).get(i).root.index){
                s=s+"Y"+","+model.trees.get(1).get(i).root.index+","+modelnew.trees.get(1).get(i).root.index+"\n";
                cY++;
                sOld.add(model.trees.get(1).get(i).root.index);
                sNew.add(modelnew.trees.get(1).get(i).root.index);
            }else{
                s=s+"N"+","+model.trees.get(1).get(i).root.index+","+modelnew.trees.get(1).get(i).root.index+"\n";
                cN++;
                sOld.add(model.trees.get(1).get(i).root.index);
                sNew.add(modelnew.trees.get(1).get(i).root.index);

            }
        }
        System.out.println(sOld.size());
        System.out.println(sNew.size());
        Set<Integer> intersection = new HashSet<Integer>(sOld);
        intersection.retainAll(sNew);
        System.out.println(intersection);
        System.out.println(intersection.size());
        s = s+"number of Y: "+cY +"; number of N: "+cN+"\n";
        s=s+"model set size: "+sOld.size()+", Feature model set size: "+sNew.size()+", intersection size: "+intersection.size()+"\n";
        File treeindexFile = new File("/Users/zhenming/Learn/Boosting/","bestIndexofTree");
        FileUtils.writeStringToFile(treeindexFile, s.toString());

    }







//    public static void main(String[] args)throws Exception {
//        List<Integer> featureIndexOld =featureindex("/Users/zhenming/Learn/Boosting/oldBoostingI800/optimized/model");
//        File oldtreeindexFile = new File("/Users/zhenming/Learn/Boosting/oldBoostingI800/optimized","bestIndexofTree");
//        FileUtils.writeStringToFile(oldtreeindexFile, featureIndexOld.toString());
//        List<Integer> featureIndexnew8 =featureindexnew("/Users/zhenming/Learn/Boosting/newBoostingI800/8/optimized/model");
//        File newtreeindexFile = new File("/Users/zhenming/Learn/Boosting/newBoostingI800/8/optimized","bestIndexofTree");
//        FileUtils.writeStringToFile(newtreeindexFile, featureIndexnew8.toString());
//
//
//    }


}

