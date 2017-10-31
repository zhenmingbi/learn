import apple.laf.JRSUIUtils;
import sun.security.util.AuthResources_pt_BR;

import javax.sound.midi.Soundbank;
import java.io.*;
import java.util.*;
/**
 * Created by zhenming on 10/16/17.
 */
public class TreeNode implements Serializable{
    private static final long serialVersionUID = 1L;
    public double val;
    public double threshold;
    public int index;
    public TreeNode left=null;
    public TreeNode right=null;
    public boolean[] bol;
    public double objective;
//    public TreeNode(double val){
//        this.val=val;
//    }


//    public static void main(String[] args){
//        TreeNode node1=new TreeNode(70.345);
//        TreeNode node2=new TreeNode(68.36);
//        TreeNode node3=new TreeNode(65.82);
//        node1.left=node2;
//        node1.right=node3;
//        Stack<TreeNode> mystack=new Stack();
//        mystack.push(node1);
//        while(!mystack.empty()){
//            TreeNode node=mystack.pop();
//            System.out.println(node.val);
//            if(node.left!=null){
//                mystack.push(node.left);
//
//            }
//            if(node.right!=null){
//                mystack.push(node.right);
//
//            }
//
//        }
//
//    }

}