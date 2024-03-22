#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <numeric>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    TreeNode* constructMaximumBinaryTree(std::vector<int>& nums) {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(NULL);

        if (nums.empty()) return nullptr;

        auto maxt = std::max_element(nums.begin(), nums.end());
        int  maxi = std::distance(nums.begin(), maxt);

        TreeNode* root = new TreeNode(*maxt);

        std::vector<int> l(nums.begin(), maxt);
        std::vector<int> r(maxt + 1, nums.end());

        root->left  = constructMaximumBinaryTree(l);
        root->right = constructMaximumBinaryTree(r);

        return root;
    }
};

int main(void) { return 0; }
