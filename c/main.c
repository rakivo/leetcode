#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

typedef struct TreeNode TreeNode;

struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

int** levelOrderBottom(struct TreeNode* root, int* returnSize, int** returnColumnSizes){
    struct TreeNode *n[2000] = { root };

    int sz = 2000, **a = malloc(sizeof(int *[sz]));
    int *c = *returnColumnSizes = malloc(sizeof(int [sz])), i = sz - 1, k;

    for (int f = 0, b = 1, *p, lb, j; root && f < b ; i--)
        for (p = a[i] = malloc(sizeof(int [c[i] = b - f])),
                lb = b, j = 0; f < lb; p[j++] = n[f++]->val)
            (n[b++] = n[f]->left) || b--, (n[b++] = n[f]->right) || b--;


    for (k = 0 ; ++i < sz ; a[k] = a[i], c[k++] = c[i]);

    return *returnSize = k, a;
}

// <=======================================================================>

typedef struct {
    int pr;
    int pf;
} p;

int cmp(const void* a, const void* b) {
    return *(int*)a - *(int*)b;
}

struct TreeNode* constructMaximumBinaryTree(int* nums, int size) {
    qsort(nums, size, sizeof(int), cmp);

    p* ps = (p*)malloc(size * sizeof(p));

    int pr = 0, pf = 0;
    for (int i = 0; i < size; i++) {
        pr += nums[i];
        pf += nums[size - 1 - i];

        ps[i].pr = pr;
        ps[i].pf = pf;
    }

    struct TreeNode* root = malloc(sizeof(TreeNode));
    root->val = nums[0];

    free(ps);

    return NULL;
}

// <=======================================================================>

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))

int levenshtein(const char* s1, const char* s2) {
    const size_t m = strlen(s1);
    const size_t n = strlen(s2);

    int prev[n + 1];
    int curr[n + 1];

    for (size_t i = 0; i <= n; ++i) {
        prev[i] = i;
    }

    for (size_t i = 0; i <= m; i++) {
        curr[0] = i;
        for (size_t j = 1; j <= n; j++) {
            if (s1[i - 1] == s2[j -  1]) {
                curr[j] = prev[j - 1];
            }
            else {
                curr[j] = 1 + min(curr[j - 1], min(prev[j], prev[j - 1]));
            }
        }
        for (size_t k = 0; k < n + 1; ++k) {
            prev[k] = curr[k];
        }
    }
    return (int) curr[n];
}

// <=======================================================================>

void preorder(struct TreeNode* root, struct TreeNode** prev) {
    if (root == NULL) return;
    struct TreeNode* left = root->left;
    struct TreeNode* right = root->right;

    if (*prev != NULL) {
        (*prev)->left = NULL;
        (*prev)->right = root;
    }
    *prev = root;

    preorder(left, prev);
    preorder(right, prev);
}

void flatten(struct TreeNode* root) {
    struct TreeNode* prev = NULL;
    preorder(root, &prev);
}

// <=======================================================================>

void perm(int* nums, int i, int* result_idx, int idx, int** result, int n) {
    if (i == n) {
        for (int j = 0; j < n; ++j) {
            result[*result_idx][j] = nums[j];
        }
        (*result_idx)++;
        return;
    }
    for (int j = i; j < n; ++j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
        perm(nums, i + 1, result_idx, idx, result, n);
        temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}

int** permute(int* nums, int n, int* ret_size, int** ret_col_size) {
    if (nums == NULL) {
        return NULL;
    }

    int total_permutations = 1;
    for (int i = 1; i <= n; ++i) {
        total_permutations *= i;
    }

    int** result = (int**)malloc(total_permutations * sizeof(int*));
    if (result == NULL) {
        return NULL;
    }
    for (int i = 0; i < total_permutations; ++i) {
        result[i] = (int*)malloc(n * sizeof(int));
    }

    int result_idx = 0;
    int idx = 0;
    perm(nums, 0, &result_idx, idx, result, n);

    *ret_col_size = (int*)malloc(total_permutations * sizeof(int));
    for (int i = 0; i < total_permutations; ++i) {
        (*ret_col_size)[i] = n;
    }

    *ret_size = total_permutations;
    return result;
}

void traverse(int* max, struct TreeNode* root) {
    if (!root) {
        return;
    }
    int r = 0, l = 0;
    if (root->right) {
        r = abs(*max - root->right->val);
    } if (root->left) {
        l = abs(*max - root->left->val);
    }
    *max = r > *max ? r : *max;
    *max = l > *max ? l : *max;
    traverse(max, root->right);
    traverse(max, root->left);
}

int maxAncestorDiff(struct TreeNode* root) {
    int ret = 0;
    traverse(&ret, root);
    return ret;
}

int main = 0;
