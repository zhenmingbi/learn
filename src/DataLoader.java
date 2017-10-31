import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import org.apache.mahout.math.Vector;


import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by zhenming on 10/22/17.
 */
public class DataLoader {
    public static double[][] loadLibsvm(String s, int numColumns)throws Exception{
        List<String> lines = Files.lines(Paths.get(s)).collect(Collectors.toList());
        double[][] x = new double[lines.size()][numColumns];
        for (int i=0;i<lines.size();i++){
            String line = lines.get(i);
            String[] newline=line.split(" ");
            for (int j=1;j<newline.length;j++){
                String[] splitByColumn=newline[j].split(":");
                x[i][Integer.parseInt(splitByColumn[0])]=Double.parseDouble(splitByColumn[1]);
            }
        }
        return x;



    }

    public static double[][] load_x(String s) throws Exception{
        List<String> lines = Files.lines(Paths.get(s)).collect(Collectors.toList());

        int columns = lines.get(0).split(",").length;
        double[][] x = new double[lines.size()][columns];
        for (int i=0;i<lines.size();i++){
            String line = lines.get(i);
            String[] newline=line.split(",");
            for (int j=0;j<columns;j++){
                x[i][j] = Double.parseDouble(newline[j]);
            }
        }
        return x;

    }

    public static void main(String[] args)throws Exception {
//        loadLibsvm("/users/zhenming/Learn/E2006.train", 150360);
        RegDataSet regDataSet = TRECFormat.loadRegDataSet("/Users/zhenming/researchData/E2006/train", DataSetType.REG_SPARSE,true);
        double[] y = regDataSet.getLabels();
        Vector firstRow = regDataSet.getRow(0);
        Vector firstColumn = regDataSet.getColumn(0);
        for (Vector.Element element: firstRow.nonZeroes()){
            int featureIndex = element.index();
            double featureValue = element.get();
            System.out.println("feature "+featureIndex+"="+featureValue);
        }

        System.out.println(firstRow.get(2));
    }
}
