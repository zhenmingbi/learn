/**
 * Created by zhenming on 10/16/17.
 */
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.stream.Collectors;

public class RegressionTree {
    TreeNode root = new TreeNode();


    private static class Result {
        int index = -1;
        double threshold = 0.0;
        double bestValue = Double.POSITIVE_INFINITY;
    }

    public static double[] getcolumn(double[][] x, int columnindex, boolean[] b) {
        double[] column = new double[bsum(b)];
        int index=0;
        for (int i = 0; i < x.length; i++) {
            if (b[i]){
                column[index] = x[i][columnindex];
                index++;

            }

        }
        return column;
    }

    public static double getMax(double[] array) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
            }
        }
        return max;
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


    public static double sum(double[] array, int l) {
        double s = 0.0;
        for (int i = 0; i < l; i++) {
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

    public static int sum(int[] array, int l) {
        int s = 0;
        for (int i = 0; i < l; i++) {
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

    public static double avg(double[] y, boolean[] b) {
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < y.length; i++) {
            if (b[i]) {
                sum += y[i];
                count += 1;
            }
        }
        return sum / count;

    }


    public Result splitnodes(double[][] x, double[] y, boolean[] b) {
        Result res = new Result();
        int total_count=bsum(b);
        double[] y_sub = new double[total_count];
        double total_sum = 0.0;
        int yindex=0;
        for (int i = 0; i < y.length; i++) {
            if (b[i]) {
                y_sub[yindex] = y[i];
                yindex++;
                total_sum += y[i];
            }
        }
        double avg_y = total_sum / total_count;
        for (int j = 0; j < x[0].length; j++) {
            double[] column = getcolumn(x, j, b);
            double jmax = getMax(column);
            double jmin = getMin(column);
            if (jmax == jmin) {
                continue;
            }
            double jincrement = (jmax - jmin) / 100.0;
            double[] jsum = new double[100];
            int[] jcount = new int[100];
            for (int i = 0; i < column.length; i++) {
                int index = (int) Math.ceil((column[i] - jmin) / jincrement) - 1;
                if (index == -1) {
                    index = 0;
                }
                if (index == 100) {
                    index = 99;
                }
                jsum[index] += y_sub[i];
                jcount[index] += 1;
            }
            double s_sum=0;
            int s_count=0;
            for (int k = 1; k < 100; k++) {
                double s = jmin + k * jincrement;
                s_sum += jsum[k-1];
                s_count += jcount[k-1];
                double objective = -(s_sum * s_sum) / s_count - ((total_sum - s_sum) * (total_sum - s_sum)) / (total_count - s_count);
                if (objective < res.bestValue) {
                    res.bestValue = objective;
                    res.index = j;
                    res.threshold = s;
                }

            }

        }
        double new_objective=-1*res.bestValue;
        for (int i = 0; i < total_count; i++) {
            new_objective+= (y_sub[i] - avg_y) * (y_sub[i] - avg_y) - y_sub[i] * y_sub[i];
        }
        res.bestValue=new_objective;
        return res;
    }

    public void fit(double[][] x, double[] y) {
        int l = x.length;
        root.bol = fill(new boolean[l]);
        root.val = avg(y, root.bol);
        Result res = splitnodes(x, y, root.bol);
        root.index = res.index;
        root.threshold = res.threshold;
        root.objective = res.bestValue;
        if (root.objective == Double.NEGATIVE_INFINITY) {
            return;
        }

        Comparator<TreeNode> comparator = Comparator.comparing(treeNode -> -1*treeNode.objective);
        PriorityQueue<TreeNode> pQueue = new PriorityQueue<TreeNode>(comparator);
        pQueue.add(root);
        int count_leaf = 1;
        while (count_leaf < 6) {
            TreeNode node = pQueue.poll();
            if (bsum(node.bol) > 5) {
                TreeNode left_child = new TreeNode();
                left_child.bol = new boolean[l];
                for (int i = 0; i < l; i++) {
                    if (x[i][node.index] <= node.threshold && node.bol[i]) {
                        left_child.bol[i] = true;

                    }
                }
                left_child.val = avg(y, left_child.bol);
                Result left_res = splitnodes(x, y, left_child.bol);
                left_child.index = left_res.index;
                left_child.threshold = left_res.threshold;
                left_child.objective = left_res.bestValue;
                node.left = left_child;
                if (left_child.objective != Double.NEGATIVE_INFINITY) {
                    pQueue.add(left_child);
                }

                TreeNode right_child = new TreeNode();
                right_child.bol = new boolean[l];
                for (int i = 0; i < l; i++) {
                    if (x[i][node.index] > node.threshold && node.bol[i]) {
                        right_child.bol[i] = true;

                    }
                }
                right_child.val = avg(y, right_child.bol);
                Result right_res = splitnodes(x, y, right_child.bol);
                right_child.index = right_res.index;
                right_child.threshold = right_res.threshold;
                right_child.objective = right_res.bestValue;
                node.right = right_child;
                if (right_child.objective != Double.NEGATIVE_INFINITY) {
                    pQueue.add(right_child);
                }
                count_leaf++;
                node.bol=null;

            }

        }
        Stack<TreeNode> mystack=new Stack<>();
        mystack.add(root);
        while(!mystack.isEmpty()){
            TreeNode node = mystack.pop();
            if(node.bol!=null){
                node.bol=null;
            }
            if(node.left!=null){
                mystack.add(node.left);
            }
            if(node.right!=null){
                mystack.add(node.right);
            }
        }

    }

    public double single_predict(double[] x) {
        TreeNode node = root;
        while (node.left != null) {
            if (x[node.index] <= node.threshold) {
                node = node.left;
            } else {
                node = node.right;

            }

        }
        return node.val;


    }

    public double[] predict(double[][]x){
        int l=x.length;
        double[] res=new double[l];
        for (int i=0; i<l;i++){
            res[i]=single_predict(x[i]);
        }
        return res;

    }

    public static double RMSE(double[] y, double[] y_predict){
        int l=y.length;
        double s=0.0;
        for(int i=0;i<l;i++){
            s+=(y[i]-y_predict[i])*(y[i]-y_predict[i]);

        }
        return Math.sqrt(s/l);

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

    public static double[] load_y(String s) throws Exception{
        List<String> lines = Files.lines(Paths.get(s)).collect(Collectors.toList());
        double[] y = new double[lines.size()];
        for (int i=0;i<lines.size();i++){
            String line = lines.get(i);
            y[i]= Double.parseDouble(line);

        }
        return y;

    }

    public void printTree(){
        Queue<TreeNode> queue = new LinkedBlockingQueue<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            System.out.println("hahahhaha");
            System.out.println(node.index);
            System.out.println(node.threshold);
            System.out.println(bsum(node.bol));
            System.out.println(node.objective);

            if (node.left != null) {
                queue.add(node.left);

            }
            if (node.right != null) {
                queue.add(node.right);

            }
        }

    }

    public static void main(String[] args)throws Exception {
        double[][] x_train=load_x("/users/zhenming/Learn/xtrain.csv");
        double[][] x_test=load_x("/users/zhenming/Learn/xtest.csv");
        double[] y_train=load_y("/users/zhenming/Learn/ytrain.csv");
        double[] y_test=load_y("/users/zhenming/Learn/ytest.csv");
        RegressionTree regressor = new RegressionTree();
        regressor.fit(x_train, y_train);
        double[] y_predict = regressor.predict(x_test);
        System.out.println(regressor.RMSE(y_test, y_predict));

   }


}







