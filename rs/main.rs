use std::rc::Rc;
use std::cell::RefCell;

#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None
        }
    }

}

macro_rules! build_tree_macro {
    ($nums:expr) => {
        match $nums.iter().enumerate().max_by_key(|&(_, val)| *val) {
            None => None,
            Some((idx, val)) => build(&$nums[..idx], &$nums[idx + 1..], *val),
        }
    };
}

pub fn construct_maximum_binary_tree(nums: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
    match nums.iter().enumerate().max_by_key(|&(_, val)| *val) {
        None => None,
        Some((idx, val)) => {
            Some(Rc::new(RefCell::new(TreeNode {
                val: *val,
                left: build_tree_macro!(&nums[..idx]),
                right: build_tree_macro!(&nums[idx + 1..]),
            })))
        }
    }
}

fn build(left: &[i32], right: &[i32], val: i32) -> Option<Rc<RefCell<TreeNode>>> {
    let mut node = TreeNode::new(val);
    node.left = build_tree_macro!(left);
    node.right = build_tree_macro!(right);
    Some(Rc::new(RefCell::new(node)))
}

// <=======================================================================>

pub fn modified_matrix(mut mat: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let (m, n) = (mat.len(), mat[0].len());
    let mut maxs = vec![0; n];

    for i in 0..m {
        for j in 0..n {
            maxs[j] = maxs[j].max(mat[i][j]);
        }
    }
    for i in 0..m {
        for j in 0..n {
            if mat[i][j] == -1 {
                mat[i][j] = maxs[j];
            }
        }
    } mat
}

fn main() {}
