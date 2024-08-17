// #include <vector>
// #include <utility>
// #include <iostream>
// #include <unordered_map>

// int maximumLength(vector<int>& nums, int k)
// {
//     size_t n = nums.size();
//     unordered_map<int, int> freq;
//     int l = 0;
//     int c = 0;
//     int ret = 0;

//     for (size_t r = 0; r < n; ++r) {
//         if (r == 0 || nums[r] != nums[r - 1]) ++c;
//         freq[nums[r]]++;
//         while (c > k + 1) {
//             if (--freq[nums[l]] == 0) freq.erase(nums[l]);
//             if (nums[l] != nums[l + 1]) --c;
//             ++l;
//         }
//         ret = max(ret, (int) r - l + 1);
//     }

//     return ret;
// }

#include <vector>
#include <ranges>
#include <utility>
#include <iostream>
#include <execution>
#include <algorithm>
#include <unordered_set>

std::vector<std::pair<int, int>> twos_difference_cxx20(const std::vector<int> &v)
{
    const std::unordered_set<int> set(v.begin(), v.end());
    std::vector<std::pair<int, int>> ret;

    for (const auto& i: v | std::views::filter([&set](const int& i) {
        return set.contains(i + 2);
    })) ret.emplace_back(i, i + 2);

    std::sort(ret.begin(), ret.end());
    return ret;
}

std::vector<std::pair<int, int>> twos_difference_cxx17(const std::vector<int> &v)
{
    std::unordered_set<int> set(v.begin(), v.end());
    std::vector<std::pair<int, int>> ret;

    for (const auto& i: v)
        if (set.count(i + 2))
            ret.emplace_back(i, i + 2);

    std::sort(std::execution::par, ret.begin(), ret.end());
    return ret;
}

std::size_t nth = 1;

void test(const std::vector<int>& v)
{
    const auto ret = twos_difference_cxx17(v);

    std::cout << "Test: #" << nth++ << std::endl;

    for (const auto& p: ret)
        std::cout << p.first << ' ' << p.second << std::endl;

    std::cout << "------" << std::endl;
}

int main(void)
{
    test({1, 2, 3, 4});
    test({4, 1, 2, 3});
    test({1, 23, 3, 4, 7});
    test({4, 3, 1, 5, 6});

    return 0;
}
