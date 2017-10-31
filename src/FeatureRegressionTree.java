import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Created by zhenming on 10/23/17.
 */
public class FeatureRegressionTree implements Serializable {
    private static final long serialVersionUID = 1L;
    TreeNode root = new TreeNode();
    private int numActiveFeatures;

    public FeatureRegressionTree(int numActiveFeatures) {
        this.numActiveFeatures = numActiveFeatures;
    }

    private static class Result {
        int index = -1;
        double threshold = 0.0;
        double bestValue = Double.POSITIVE_INFINITY;
    }

    private static class NonzeroAttributes {
        double max;
        int countnonzerosTrue;

    }


    public static NonzeroAttributes getMax(Vector column, boolean[] b, int bsum) {
        NonzeroAttributes nza = new NonzeroAttributes();
        nza.max = Double.NEGATIVE_INFINITY;
        nza.countnonzerosTrue = 0;
        for (Vector.Element element : column.nonZeroes()) {
            double value = element.get();
            int index = element.index();
            if (b[index]) {
                nza.countnonzerosTrue += 1;
                if (value > nza.max) {
                    nza.max = value;
                }
            }

        }
        if (nza.countnonzerosTrue != bsum && nza.max < 0) {
            nza.max = 0;
        }

        return nza;
    }


    public static double getMin(Vector column, boolean[] b, int bsum) {
        double min = Double.POSITIVE_INFINITY;
        int countNonzerosTrue = 0;
        for (Vector.Element element : column.nonZeroes()) {
            double value = element.get();
            int index = element.index();
            if (b[index]) {
                countNonzerosTrue += 1;
                if (value < min) {
                    min = value;
                }
            }

        }
        if (countNonzerosTrue != bsum && min > 0) {
            return 0;
        }

        return min;
    }

    public static double avg(double[] y, boolean[] b) {
        double s = 0;
        int l = 0;
        for (int i = 0; i < y.length; i++) {
            if (b[i]) {
                s += y[i];
                l+=1;
            }
        }
        return s / l;

    }

    public PriorityQueue<Result> rootsplit(DataSet x, double[] y, boolean[] b) {
        int total_count = MyMath.bsum(b);
        double[] y_sub = new double[total_count];
        double total_sum = 0.0;
        int yindex = 0;
        for (int i = 0; i < y.length; i++) {
            if (b[i]) {
                y_sub[yindex] = y[i];
                yindex++;
                total_sum += y[i];
            }
        }
        Comparator<Result> comparator = Comparator.comparing(Result -> Result.bestValue);
        PriorityQueue<Result> fQueue = new PriorityQueue<>(comparator);

        // TODO: full scan
        for (int j = 0; j < x.getNumFeatures(); j++) {
            Result res = new Result();
            res.index=j;
            Vector column = x.getColumn(j);
            NonzeroAttributes nza = getMax(column, b, total_count);
            double jmax = nza.max;
            int numofNonzero = nza.countnonzerosTrue;
            double jmin = getMin(column, b, total_count);
            if (jmax == jmin) {
                continue;
            }
            double jincrement = (jmax - jmin) / 100.0;
            double[] jsum = new double[100];
            int[] jcount = new int[100];
            double nonzeroSum = 0;
            for (Vector.Element element : column.nonZeroes()) {
                if(b[element.index()]){
                    double elementValue = element.get();
                    nonzeroSum += y[element.index()];

                    int index = (int) Math.ceil((elementValue - jmin) / jincrement) - 1;
                    if (index == -1) {
                        index = 0;
                    }
                    if (index == 100) {
                        index = 99;
                    }
                    jsum[index] += y[element.index()];
                    jcount[index] += 1;

                }

            }
            if (numofNonzero< total_count) {
                int zeroindex = (int) Math.ceil((0 - jmin) / jincrement) - 1;
                if (zeroindex == -1) {
                    zeroindex = 0;
                }
                if (zeroindex == 100) {
                    zeroindex = 99;
                }
                jsum[zeroindex] += total_sum - nonzeroSum;
                jcount[zeroindex] += total_count - numofNonzero;
            }
            double s_sum = 0;
            int s_count = 0;
            for (int k = 1; k < 100; k++) {
                double s = jmin + k * jincrement;
                s_sum += jsum[k - 1];
                s_count += jcount[k - 1];
                double objective = -(s_sum * s_sum) / s_count - ((total_sum - s_sum) * (total_sum - s_sum)) / (total_count - s_count);
                if (objective < res.bestValue) {
                    res.bestValue = objective;
                    res.threshold = s;
                }
            }
            fQueue.add(res);

        }
        return fQueue;
    }




    public Result splitnodes(DataSet x, double[] y, boolean[] b, List<Integer>flist) {
        Result res = new Result();
        int total_count = MyMath.bsum(b);
        double[] y_sub = new double[total_count];
        double total_sum = 0.0;
        int yindex = 0;
        for (int i = 0; i < y.length; i++) {
            if (b[i]) {
                y_sub[yindex] = y[i];
                yindex++;
                total_sum += y[i];
            }
        }
        double avg_y = total_sum / total_count;
        // TODO: local scan
        for (int j:flist) {
            Vector column = x.getColumn(j);
            NonzeroAttributes nza = getMax(column, b, total_count);
            double jmax = nza.max;
            int numofNonzero = nza.countnonzerosTrue;
            double jmin = getMin(column, b, total_count);
            if (jmax == jmin) {
                continue;
            }
            double jincrement = (jmax - jmin) / 100.0;
            double[] jsum = new double[100];
            int[] jcount = new int[100];
            double nonzeroSum = 0;
            for (Vector.Element element : column.nonZeroes()) {
                if(b[element.index()]){
                    double elementValue = element.get();
                    nonzeroSum += y[element.index()];

                    int index = (int) Math.ceil((elementValue - jmin) / jincrement) - 1;
                    if (index == -1) {
                        index = 0;
                    }
                    if (index == 100) {
                        index = 99;
                    }
                    jsum[index] += y[element.index()];
                    jcount[index] += 1;

                }

            }
            if (numofNonzero< total_count) {
                int zeroindex = (int) Math.ceil((0 - jmin) / jincrement) - 1;
                if (zeroindex == -1) {
                    zeroindex = 0;
                }
                if (zeroindex == 100) {
                    zeroindex = 99;
                }
                jsum[zeroindex] += total_sum - nonzeroSum;
                jcount[zeroindex] += total_count - numofNonzero;
            }
            double s_sum = 0;
            int s_count = 0;
            for (int k = 1; k < 100; k++) {
                double s = jmin + k * jincrement;
                s_sum += jsum[k - 1];
                s_count += jcount[k - 1];
                double objective = -(s_sum * s_sum) / s_count - ((total_sum - s_sum) * (total_sum - s_sum)) / (total_count - s_count);
                if (objective < res.bestValue) {
                    res.bestValue = objective;
                    res.index = j;
                    res.threshold = s;
                }

            }

        }
        double new_objective = -1 * res.bestValue;
        for (int i = 0; i < total_count; i++) {
            new_objective += (y_sub[i] - avg_y) * (y_sub[i] - avg_y) - y_sub[i] * y_sub[i];
        }
        res.bestValue = new_objective;
        return res;
    }

    public void fit(DataSet x, double[] y, List<Integer>flist, int numofLeaf) {
        int l = y.length;
        root.bol = new boolean[l];
        Arrays.fill(root.bol, true);
        root.val = avg(y, root.bol);
        Result res = splitnodes(x, y, root.bol, flist);
        root.index = res.index;
        root.threshold = res.threshold;
        root.objective = res.bestValue;
        if (root.objective == Double.NEGATIVE_INFINITY) {
            return;
        }

        Comparator<TreeNode> comparator = Comparator.comparing(treeNode -> -1 * treeNode.objective);
        PriorityQueue<TreeNode> pQueue = new PriorityQueue<TreeNode>(comparator);
        pQueue.add(root);
        int count_leaf = 1;
        while (count_leaf < numofLeaf) {
            TreeNode node = pQueue.poll();
            if (MyMath.bsum(node.bol) > 5) {
                TreeNode left_child = new TreeNode();
                left_child.bol = new boolean[l];
                TreeNode right_child = new TreeNode();
                right_child.bol = new boolean[l];
                Vector column = x.getColumn(node.index);
                Vector densecolumn = new DenseVector(column);
                for (int i = 0; i < l; i++) {
                    if (node.bol[i]) {
                        if (densecolumn.get(i) <= node.threshold) {
                            left_child.bol[i] = true;
                        } else {
                            right_child.bol[i] = true;
                        }
                    }
                }
                left_child.val = avg(y, left_child.bol);
                right_child.val = avg(y, right_child.bol);
                node.left = left_child;
                node.right = right_child;
                if(count_leaf<numofLeaf-1){
                    Result left_res = splitnodes(x, y, left_child.bol, flist);
                    left_child.index = left_res.index;
                    left_child.threshold = left_res.threshold;
                    left_child.objective = left_res.bestValue;
                    if (left_child.objective != Double.NEGATIVE_INFINITY) {
                        pQueue.add(left_child);
                    }
                    Result right_res = splitnodes(x, y, right_child.bol, flist);
                    right_child.index = right_res.index;
                    right_child.threshold = right_res.threshold;
                    right_child.objective = right_res.bestValue;

                    if (right_child.objective != Double.NEGATIVE_INFINITY) {
                        pQueue.add(right_child);
                    }
                }
                count_leaf++;
                node.bol = null;

            }

        }
        Stack<TreeNode> mystack = new Stack<>();
        mystack.add(root);
        while (!mystack.isEmpty()) {
            TreeNode node = mystack.pop();
            if (node.bol != null) {
                node.bol = null;
            }
            if (node.left != null) {
                mystack.add(node.left);
            }
            if (node.right != null) {
                mystack.add(node.right);
            }
        }

    }

    public List<Integer> fit(DataSet x, double[] y, int numofLeaf) {
        int l = y.length;
        root.bol = new boolean[l];
        Arrays.fill(root.bol, true);
        root.val = avg(y, root.bol);
        PriorityQueue<Result> fQueue = rootsplit(x, y, root.bol);
        Result res=fQueue.peek();
        List<Integer>flist=new ArrayList<>();
        for(int i=0; i<numActiveFeatures;i++){
            Result result=fQueue.poll();
            flist.add(result.index);
        }
        root.index = res.index;
        root.threshold = res.threshold;
        double new_objective = -1 * res.bestValue;
        for (int i = 0; i < y.length; i++) {
            new_objective += (y[i] - root.val) * (y[i] - root.val) - y[i] * y[i];
        }
        res.bestValue = new_objective;
        root.objective = res.bestValue;
        if (root.objective == Double.NEGATIVE_INFINITY) {
            return flist;
        }


        Comparator<TreeNode> comparator = Comparator.comparing(treeNode -> -1 * treeNode.objective);
        PriorityQueue<TreeNode> pQueue = new PriorityQueue<TreeNode>(comparator);
        pQueue.add(root);
        int count_leaf = 1;
        while (count_leaf < numofLeaf) {
            TreeNode node = pQueue.poll();
            if (MyMath.bsum(node.bol) > 5) {
                TreeNode left_child = new TreeNode();
                left_child.bol = new boolean[l];
                TreeNode right_child = new TreeNode();
                right_child.bol = new boolean[l];
                Vector column = x.getColumn(node.index);
                Vector densecolumn = new DenseVector(column);
                for (int i = 0; i < l; i++) {
                    if (node.bol[i]) {
                        if (densecolumn.get(i) <= node.threshold) {
                            left_child.bol[i] = true;
                        } else {
                            right_child.bol[i] = true;
                        }
                    }
                }
                left_child.val = avg(y, left_child.bol);
                right_child.val = avg(y, right_child.bol);
                node.left = left_child;
                node.right = right_child;
                if(count_leaf<numofLeaf-1){
                    Result left_res = splitnodes(x, y, left_child.bol, flist);
                    left_child.index = left_res.index;
                    left_child.threshold = left_res.threshold;
                    left_child.objective = left_res.bestValue;

                    if (left_child.objective != Double.NEGATIVE_INFINITY) {
                        pQueue.add(left_child);
                    }
                    Result right_res = splitnodes(x, y, right_child.bol, flist);
                    right_child.index = right_res.index;
                    right_child.threshold = right_res.threshold;
                    right_child.objective = right_res.bestValue;
                    if (right_child.objective != Double.NEGATIVE_INFINITY) {
                        pQueue.add(right_child);
                    }
                }

                count_leaf++;
                node.bol = null;

            }

        }
        Stack<TreeNode> mystack = new Stack<>();
        mystack.add(root);
        while (!mystack.isEmpty()) {
            TreeNode node = mystack.pop();
            if (node.bol != null) {
                node.bol = null;
            }
            if (node.left != null) {
                mystack.add(node.left);
            }
            if (node.right != null) {
                mystack.add(node.right);
            }
        }
        return flist;

    }

    public double single_predict(Vector x) {
        TreeNode node = root;
        while (node.left != null) {
            if (x.get(node.index) <= node.threshold) {
                node = node.left;
            } else {
                node = node.right;

            }

        }
        return node.val;


    }

    public double[] predict(DataSet x) {
        int l = x.getNumDataPoints();
        double[] res = new double[l];
        for (int i = 0; i < l; i++) {
            res[i] = single_predict(x.getRow(i));
        }
        return res;

    }

    public static double RMSE(double[] y, double[] y_predict) {
        int l = y.length;
        double s = 0.0;
        for (int i = 0; i < l; i++) {
            s += (y[i] - y_predict[i]) * (y[i] - y_predict[i]);

        }
        return Math.sqrt(s / l);

    }


    public void printTree() {
        Queue<TreeNode> queue = new LinkedBlockingQueue<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            System.out.println("hahahhaha");
            System.out.println(node.index);
            System.out.println(node.threshold);
            System.out.println(node.val);

            if (node.left != null) {
                queue.add(node.left);

            }
            if (node.right != null) {
                queue.add(node.right);

            }
        }

    }

    public static void main(String[] args) throws Exception {
        RegDataSet x_train = TRECFormat.loadRegDataSet("/Users/zhenming/researchData/E2006/train", DataSetType.REG_SPARSE, true);
        double[] y_train = x_train.getLabels();
        RegDataSetRegressionTree regressor = new RegDataSetRegressionTree();
        regressor.fit(x_train, y_train,2);
        x_train = null;
//        RegDataSet x_test=TRECFormat.loadRegDataSet("/Users/zhenming/Learn/E2006/test", DataSetType.REG_SPARSE, true)
//        double[] y_test=x_test.getLabels();
//        System.out.println(regressor.predict(x_test));
        regressor.printTree();
    }


}
