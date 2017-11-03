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
        String t="";
        GradientBoostingClassification model = (GradientBoostingClassification) Serialization.deserialize("/Users/zhenming/Learn/Boosting/amazoncat-13k/oldBoostingI800/optimized/model");
        FeatureBoostingClassification modelnew = (FeatureBoostingClassification) Serialization.deserialize("/Users/zhenming/Learn/Boosting/amazoncat-13k/newBoostingI800/8/optimized/model");
        int cY=0;
        int cN=0;
        for(int i=0; i<800;i++){
            if(model.trees.get(1).get(i).root.index == modelnew.trees.get(1).get(i).root.index){
                s=s+"Y"+","+model.trees.get(1).get(i).root.index+","+modelnew.trees.get(1).get(i).root.index+"\n";
                t=t+"="+","+model.trees.get(1).get(i).root.objective+","+modelnew.trees.get(1).get(i).root.objective+"\n";
                cY++;
                sOld.add(model.trees.get(1).get(i).root.index);
                sNew.add(modelnew.trees.get(1).get(i).root.index);
            }else{
                s=s+"N"+","+model.trees.get(1).get(i).root.index+","+modelnew.trees.get(1).get(i).root.index+"\n";
                t=t+"#"+","+model.trees.get(1).get(i).root.objective+","+modelnew.trees.get(1).get(i).root.objective+"\n";
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
        File treeobjFile = new File("/Users/zhenming/Learn/Boosting/","objectiveofTreeroot");
        FileUtils.writeStringToFile(treeobjFile, t.toString());

        Set<Integer> s_resOld=new HashSet<>();
        Set<Integer> s_resNew=new HashSet<>();
        String s_res="";
        String t_res="";
        RegDataSetGradientBoosting res_model = (RegDataSetGradientBoosting) Serialization.deserialize("/Users/zhenming/Learn/Boosting/e2006-log1p/oldBoosting/model");
        FeatureGradientBoosting res_modelnew = (FeatureGradientBoosting) Serialization.deserialize("/Users/zhenming/Learn/Boosting/e2006-log1p/newBoosting/fea10/10/model");
        int c_resY=0;
        int c_resN=0;
        for(int i=0; i<200;i++){
            if(res_model.trees.get(i).root.index == res_modelnew.trees.get(i).root.index){
                s_res=s_res+"Y"+","+res_model.trees.get(i).root.index+","+res_modelnew.trees.get(i).root.index+"\n";
                t_res=t_res+"="+","+res_model.trees.get(i).root.objective+","+res_modelnew.trees.get(i).root.objective+"\n";
                c_resY++;
                s_resOld.add(res_model.trees.get(i).root.index);
                s_resNew.add(res_modelnew.trees.get(i).root.index);
            }else{
                s_res=s_res+"N"+","+res_model.trees.get(i).root.index+","+res_modelnew.trees.get(i).root.index+"\n";
                t_res=t_res+"#"+","+res_model.trees.get(i).root.objective+","+res_modelnew.trees.get(i).root.objective+"\n";
                c_resN++;
                s_resOld.add(res_model.trees.get(i).root.index);
                s_resNew.add(res_modelnew.trees.get(i).root.index);

            }
        }
        System.out.println(s_resOld.size());
        System.out.println(s_resNew.size());
        Set<Integer> res_intersection = new HashSet<Integer>(s_resOld);
        res_intersection.retainAll(s_resNew);
        System.out.println(res_intersection);
        System.out.println(res_intersection.size());
        System.out.println(s_resOld.containsAll(s_resNew));
        s_res = s_res+"number of Y: "+c_resY +"; number of N: "+c_resN+"\n";
        s_res=s_res+"model set size: "+s_resOld.size()+", Feature model set size: "+s_resNew.size()+", intersection size: "+res_intersection.size()+"\n";
        File restreeindexFile = new File("/Users/zhenming/Learn/Boosting/","bestIndexofResTree");
        FileUtils.writeStringToFile(restreeindexFile, s_res.toString());
        File restreeobjFile = new File("/Users/zhenming/Learn/Boosting/","objectiveofResTreeroot");
        FileUtils.writeStringToFile(restreeobjFile, t_res.toString());


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

