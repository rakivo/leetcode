// Link to my leet code: https://leetcode.com/u/marktyrkba/
pub fn num_islands(mut grid: Vec<Vec<char>>) -> i32 {
    //                               l        r       u        d
    const DIRS: [(i32, i32); 4] = [(0, -1), (0, 1), (1, 0), (-1, 0)];
    fn dfs(i: i32, j: i32, grid: &mut Vec<Vec<char>>, n: &i32, n0: &i32) -> bool {
        if i < 0 || j < 0 || i >= *n || j >= *n0 || grid[i as usize][j as usize] == '0' {
            return false;
        }
        grid[i as usize][j as usize] = '0';
        for (di, dj) in DIRS {
            dfs(i + di, j + dj, grid, n, n0);
        }
        true
    }
    let n = grid.len() as i32;
    let n0 = grid[0].len() as i32;
    let mut ans = 0;

    for i in 0..n {
        for j in 0..n0 {
            if dfs(i, j, &mut grid, &n, &n0) {
                ans += 1;
            }
        }
    }
    ans
}

// <=======================================================================>

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}

fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        if b < a {
            std::mem::swap(&mut b, &mut a);
        }
        b %= a;
    }
    a
}

pub fn insert_greatest_common_divisors(mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut current = &mut head;

    while let Some(node) = current {
        if let Some(next_node) = node.next.take() {
            let mut new_node = Box::new(ListNode::new(gcd(node.val, next_node.val)));
            new_node.next = Some(next_node);

            node.next = Some(new_node);
            current = &mut node.next.as_mut().unwrap().next;
        } else {
            break;
        }
    }
    head
}

// <=======================================================================>

use std::cell::RefCell;
use std::rc::Rc;

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
            right: None,
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
        Some((idx, val)) => Some(Rc::new(RefCell::new(TreeNode {
            val: *val,
            left: build_tree_macro!(&nums[..idx]),
            right: build_tree_macro!(&nums[idx + 1..]),
        }))),
    }
}

fn build(left: &[i32], right: &[i32], val: i32) -> Option<Rc<RefCell<TreeNode>>> {
    let mut node = TreeNode::new(val);
    node.left = build_tree_macro!(left);
    node.right = build_tree_macro!(right);
    Some(Rc::new(RefCell::new(node)))
}

// <=============================================================================>

#[allow(unused)]
struct RandomizedSet {
    n: usize,
    state: usize,
    map: HashMap<i32, usize>,
    values: [Option<i32>; 5000],
}

#[allow(unused)]
impl RandomizedSet {
    fn new() -> RandomizedSet {
        RandomizedSet {
            n: 0,
            state: 1488,
            map: HashMap::with_capacity(200000),
            values: [None; 5000],
        }
    }

    fn insert(&mut self, val: i32) -> bool {
        if self.map.contains_key(&val) {
            return false;
        }
        self.values[self.n] = Some(val);
        self.map.insert(val, self.n);
        self.n += 1;
        true
    }

    fn remove(&mut self, val: i32) -> bool {
        if let Some(index) = self.map.remove(&val) {
            let last = self.values[self.n - 1].unwrap();
            self.values[index] = Some(last);
            self.map.insert(last, index);

            self.values[self.n - 1] = None;
            self.map.remove(&val);
            self.n -= 1;
            true
        } else {
            false
        }
    }

    fn get_random(&mut self) -> i32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.values[self.state % self.n].unwrap()
    }
}

// <=============================================================================>

use std::collections::HashSet;

struct RandomizedCollection {
    map: HashMap<i32, (HashSet<usize>, usize)>,
    values: [Option<i32>; 10000],
    state: usize,
    n: usize,
}

#[allow(unused)]
impl RandomizedCollection {
    fn new() -> RandomizedCollection {
        RandomizedCollection {
            map: HashMap::new(),
            values: [None; 10000],
            state: 1488,
            n: 0,
        }
    }

    fn insert(&mut self, val: i32) -> bool {
        self.values[self.n] = Some(val);
        self.n += 1;

        let (set, n) = self.map.entry(val).or_insert_with(|| (HashSet::new(), 0));
        set.insert(self.n - 1);
        *n += 1;
        *n - 1 == 0
    }

    fn remove(&mut self, val: i32) -> bool {
        if let Some((set, n)) = self.map.get_mut(&val) {
            *n -= 1;
            let i = *set.iter().next().unwrap();
            let end = self.n - 1;

            let curr = self.values[i].unwrap();
            let last = self.values[end].unwrap();
            let setb = &mut self.map.get_mut(&last).unwrap().0;
            if curr != last {
                setb.remove(&end);
                setb.insert(i);
                self.map.get_mut(&curr).unwrap().0.remove(&i);
            } else {
                setb.remove(&end);
            }

            self.values.swap(i, end);
            self.n -= 1;
            if self.map.get_mut(&curr).unwrap().1 == 0 {
                self.map.remove(&curr);
            }
            true
        } else {
            false
        }
    }

    fn get_random(&mut self) -> i32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.values[self.state % self.n].unwrap()
    }
}

// <=============================================================================>

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
    }
    mat
}

// <=======================================================================>

// only because of that problem is for "LeetCode subscribers" and closed for other people
#[allow(unused)]
fn meeting_rooms(mut times: Vec<Vec<u32>>) -> bool {
    times.sort_by(|i1, i2| i1[0].cmp(&i2[0]));
    times.windows(2).all(|vec| vec[0][1] <= vec[1][0])
}

// <=======================================================================>

pub fn count_prefix_suffix_pairs(words: Vec<String>) -> i64 {
    words
        .iter()
        .fold(
            (0, HashMap::<&str, (i64, usize)>::new()),
            |(mut acc, mut map), w| {
                let n = w.len();
                map.iter().for_each(|(k, (v, len))| {
                    if *len <= n && w.starts_with(k) && w.ends_with(k) {
                        acc += v;
                    }
                });
                map.entry(w.as_str()).or_insert_with(|| (0, n)).0 += 1;
                (acc, map)
            },
        )
        .0
}

// <=======================================================================>

//little easy problem to have chill a little minute
pub fn map_product_difference(mut nums: Vec<i32>) -> i32 {
    let n = nums.len();
    nums.sort_unstable();
    (nums[n - 1] * nums[n - 2]) - (nums[0] * nums[1])
}

// <=======================================================================>

pub fn count_consistent_strings(allowed: String, words: Vec<String>) -> i32 {
    words
        .iter()
        .filter(|w| w.chars().all(|ch| allowed.contains(ch)))
        .count() as i32
}

// <=======================================================================>

pub fn is_possible_to_split(nums: Vec<i32>) -> bool {
    let mut map: [u8; 101] = [0; 101];
    nums.iter().all(|i| {
        let i = *i as usize;
        map[i] += 1;
        map[i] <= 2
    })
}

pub fn result_array1(nums: Vec<i32>) -> Vec<i32> {
    let mut arr1 = vec![nums[0]];
    let mut arr2 = vec![nums[1]];

    nums.iter().skip(2).for_each(|&i| {
        if arr1.last() > arr2.last() {
            arr1.push(i);
        } else {
            arr2.push(i);
        }
    });
    arr1.extend(arr2);
    arr1
}

pub fn result_arrayfold(nums: Vec<i32>) -> Vec<i32> {
    let (mut arr1, arr2) = nums
        .iter()
        .skip(2)
        .fold(
            ((vec![nums[0]], vec![nums[1]]), (0, 0)),
            |((mut arr1, mut arr2), (mut n1, mut n2)), &i| {
                if arr1[n1] > arr2[n2] {
                    arr1.push(i);
                    n1 += 1;
                } else {
                    arr2.push(i);
                    n2 += 1;
                }
                ((arr1, arr2), (n1, n2))
            },
        )
        .0;
    arr1.extend(arr2);
    arr1
}

// <=======================================================================>

#[allow(unused)]
fn find_and_replace_pattern(words: Vec<String>, pattern: String) -> Vec<String> {
    fn is_match(word: &str, pattern: &str) -> bool {
        let mut w2pmap = HashMap::new();
        let mut p2wmap = HashMap::new();

        let ch_word = word.chars().collect::<Vec<_>>();
        let ch_patt = pattern.chars().collect::<Vec<_>>();

        for i in 0..ch_word.len() {
            if let Some(&prevw) = p2wmap.get(&ch_patt[i]) {
                if prevw != ch_word[i] {
                    return false;
                }
            } else {
                p2wmap.insert(ch_patt[i], ch_word[i]);
            }

            if let Some(&prevp) = w2pmap.get(&ch_word[i]) {
                if prevp != ch_patt[i] {
                    return false;
                }
            } else {
                w2pmap.insert(ch_word[i], ch_patt[i]);
            }
        }
        true
    }

    words
        .into_iter()
        .filter(|word| is_match(word, &pattern))
        .collect()
}

// <=======================================================================>

pub fn can_be_typed_words(text: String, bale: String) -> i32 {
    text.split_whitespace()
        .filter(|w| bale.chars().all(|c| !w.contains(c)))
        .count() as i32
}

// <=======================================================================>

pub fn count_prefixes(words: Vec<String>, s: String) -> i32 {
    words.iter().filter(|&prefix| s.starts_with(prefix)).count() as i32
}

// <=======================================================================>

#[allow(unused)]
fn gcd1(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

#[allow(unused)]
fn solution(arr: &[u64]) -> u128 {
    arr.iter().fold(arr[0], |acc, &x| gcd1(acc, x)) as u128 * arr.len() as u128
}

// <=======================================================================>

struct FT {
    tree: Vec<usize>,
    treen: i32,
}

impl FT {
    fn new(max: usize) -> FT {
        FT {
            tree: vec![0; max + 2],
            treen: (max + 2) as i32,
        }
    }

    fn update(&mut self, mut pos: i32, val: usize) {
        while pos < self.treen {
            self.tree[pos as usize] += val;
            pos += pos & -pos;
        }
    }

    fn sum(&self, mut pos: i32) -> usize {
        let mut ans = 0;
        while pos > 0 {
            ans += self.tree[pos as usize];
            pos -= pos & -pos;
        }
        ans
    }
}

macro_rules! push {
    ($arr: ident <- $i: expr, $n: ident++, $ft: ident <- $id: ident) => {
        $arr.push($i);
        $n += 1;
        $ft.update($id, 1);
    };
}

pub fn result_array(nums: Vec<i32>) -> Vec<i32> {
    let n = nums.len();

    let (mut arr1, mut arr2) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut n1, mut n2) = (1, 1);
    let (mut map, mut mapn) = (HashMap::with_capacity(n), 0);

    let mut sorted = nums.clone();
    sorted.sort_unstable();
    sorted.iter().for_each(|i| {
        map.entry(i).or_insert(mapn + 1);
        mapn += 1;
    });

    let (mut ft1, mut ft2) = (FT::new(mapn as usize), FT::new(mapn as usize));
    arr1.push(nums[0]);
    ft1.update(*map.get(&nums[0]).unwrap(), 1);

    arr2.push(nums[1]);
    ft2.update(*map.get(&nums[1]).unwrap(), 1);
    nums.iter().skip(2).for_each(|&i| {
        let id = *map.get(&i).unwrap();
        let gc1 = n1 - ft1.sum(id);
        let gc2 = n2 - ft2.sum(id);
        if gc1 > gc2 {
            push!(arr1 <- i, n1++, ft1 <- id);
        } else if gc1 < gc2 {
            push!(arr2 <- i, n2++, ft2 <- id);
        } else {
            if n2 < n1 {
                push!(arr2 <- i, n2++, ft2 <- id);
            } else {
                push!(arr1 <- i, n1++, ft1 <- id);
            }
        }
    });
    arr1.append(&mut arr2);
    arr1
}

// <=======================================================================>

pub fn beautiful_substrings(s: String, k: i32) -> i32 {
    let n = s.len();
    let mut dp = vec![(0, 0); n];

    for (i, c) in s.chars().enumerate() {
        dp[i] = if i == 0 { (0, 0) } else { dp[i - 1] };

        if "aeiou".contains(c) {
            dp[i].0 += 1;
        } else {
            dp[i].1 += 1;
        }
    }

    let mut t = 0;
    for i in 0..n {
        if dp[i].0 == dp[i].1 && (dp[i].0 * dp[i].1) % k == 0 {
            t += 1;
        }

        for j in 0..i {
            let v = dp[i].0 - dp[j].0;
            let c = dp[i].1 - dp[j].1;

            if v == c && (v * c) % k == 0 {
                t += 1;
            }
        }
    }
    t
}

// <=======================================================================>

pub fn sum_of_encrypted_int(n: Vec<i32>) -> i32 {
    n.iter()
        .map(|&x| {
            let mut len = 0;
            let mut max = 0;
            let mut y = x;
            while y > 0 {
                max = std::cmp::max(y % 10, max);
                y /= 10;
                len += 1;
            }
            max * (10i32.pow(len) - 1) / 9
        })
        .sum()
}

// <=======================================================================>

// ...
pub fn is_substring_present(s: String) -> bool {
    let x = s.chars().rev().collect::<String>();
    for i in 0..s.len() - 1 {
        if x.contains(&s[i..i + 2]) {
            return true;
        }
    }
    false
}

// <=======================================================================>

// mine (not posted)
pub fn is_substring_present1(s: String) -> bool {
    (0..s.len() - 1).any(|i| s.chars().rev().collect::<String>().contains(&s[i..i + 2]))
}

// <=======================================================================>

// author: https://leetcode.com/problems/minimum-deletions-to-make-string-k-special/solutions/4886256/o-n-solution-rust
pub fn find_minimum_operations(s1: String, s2: String, s3: String) -> i32 {
    let (x, y, z) = (s1.len(), s2.len(), s3.len());
    let (mut i, m) = (0, x.min(y).min(z));

    while i < m && &s1[i..=i] == &s2[i..=i] && &s2[i..=i] == &s3[i..=i] {
        i += 1
    }

    match i {
        0 => -1,
        _ => (x + y + z - 3 * i) as i32,
    }
}

// <=======================================================================>

// mine (not posted)
pub fn count_substrings(s: String, c: char) -> i64 {
    let m = s.chars().filter(|&ch| ch == c).count() as i64;
    m * (m + 1) / 2
}

// <=======================================================================>

// don't remember
pub fn minimum_abs_difference(mut arra: Vec<i32>) -> Vec<Vec<i32>> {
    arra.sort_unstable();
    let m = arra.windows(2).map(|p| p[1] - p[0]).min().unwrap();
    arra.windows(2)
        .filter_map(|p| {
            if p[1] - p[0] == m {
                Some(vec![p[0], p[1]])
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
}

// <=======================================================================>

// mine (not posted)
pub fn find_max_k(mut nums: Vec<i32>) -> i32 {
    nums.sort_unstable();
    nums.iter()
        .rev()
        .filter_map(|i| {
            if nums.binary_search(&-i).is_ok() {
                Some(*i)
            } else {
                None
            }
        })
        .max()
        .unwrap_or(-1)
}

// <=======================================================================>

// mine (not posted)
pub fn search(nums: Vec<i32>, tar: i32) -> i32 {
    let at = nums.partition_point(|&x| x >= nums[0]);
    if tar < nums[0] {
        if let Some(idx) = nums[at..].binary_search(&tar).ok() {
            return (idx + at) as i32;
        }
    } else {
        if let Some(idx) = nums[..at].binary_search(&tar).ok() {
            return idx as i32;
        }
    }
    -1
}

// <=======================================================================>

// author: https://leetcode.com/problems/minimum-deletions-to-make-string-k-special/solutions/4886256/o-n-solution-rust
pub fn minimum_deletions(word: String, k: i32) -> i32 {
    let mut map = [0; 26];
    let mut ans = i32::MAX;

    for c in word.chars() {
        map[c as usize - 97] += 1;
    }

    for &x in map.iter() {
        let high = x + k;
        ans = ans.min(
            map.iter()
                .map(|&y| {
                    if y > high {
                        y - high
                    } else if y < x {
                        y
                    } else {
                        0
                    }
                })
                .sum(),
        );
    }
    ans
}

// <=======================================================================>

// author: https://leetcode.com/problems/game-of-life/solutions/3495635/o-1-space-in-place-solution-with-bit-manipulation
pub fn game_of_life(a: &mut Vec<Vec<i32>>) {
    let n = a.len();
    let m = a[0].len();

    for i in 0..n {
        for j in 0..m {
            let mut live_cnt = 0;
            for di in [!0, 0, 1] {
                for dj in [!0, 0, 1] {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    if i.wrapping_add(di) >= n || j.wrapping_add(dj) >= m {
                        continue;
                    }
                    live_cnt += a[i + di][j + dj] & 1;
                }
            }
            if a[i][j] % 2 == 1 && (live_cnt == 2 || live_cnt == 3) {
                a[i][j] |= 1 << 1;
            }
            if a[i][j] % 2 == 0 && live_cnt == 3 {
                a[i][j] |= 1 << 1;
            }
        }
    }
    for i in 0..n {
        for j in 0..m {
            a[i][j] >>= 1;
        }
    }
}

use std::collections::HashMap;

// <=======================================================================>

// mine: https://leetcode.com/problems/maximum-length-substring-with-two-occurrences/solutions/4919093/0ms-one-liner-beats-100-spaces-runtimes-the-fastest-in-the-entire-world-btw-one-liner-xd
#[allow(unused)]
fn maximum_length_substring(s: String) -> i32 {
    s.chars()
        .enumerate()
        .fold(
            (std::collections::HashMap::<u8, usize>::new(), 0, 0),
            |(mut f, a, mut j), (i, c)| {
                *f.entry(c as u8).or_insert(0) += 1;
                while j < i && f[&(c as u8)] > 2 {
                    if let Some(cnt) = f.get_mut(&s.as_bytes()[j]) {
                        *cnt -= 1;
                    }
                    j += 1;
                }
                (f, a.max(i - j + 1), j)
            },
        )
        .1 as i32
}

// <=======================================================================>

// mine (not posted)
pub fn return_to_boundary_count(nums: Vec<i32>) -> i32 {
    nums.iter()
        .fold((0, 0), |(mut r, mut c), i| {
            c += i;
            if c == 0 {
                r += 1;
            }
            (r, c)
        })
        .0
}

// <=======================================================================>

// author: https://leetcode.com/problems/lexicographical-numbers/solutions/2969038/100-faster-solution-in-rust
pub fn lexical_order(n: i32) -> Vec<i32> {
    let mut res: Vec<i32> = Vec::new();

    fn dfs(cur: i32, n: i32, res: &mut Vec<i32>) {
        if cur > n {
            return;
        }

        res.push(cur);
        dfs(cur * 10, n, res);
        if cur % 10 != 9 {
            dfs(cur + 1, n, res);
        }
    }

    dfs(1, n, &mut res);
    res
}

// <=======================================================================>

pub fn decrypt(code: Vec<i32>, k: i32) -> Vec<i32> {
    let len = code.len() as i32;
    match k {
        0 => vec![0; len as usize],
        _ => {
            if k < 0 {
                (0..len)
                    .map(|index| {
                        (index + k..index)
                            .map(|i| code[i.rem_euclid(len) as usize])
                            .sum()
                    })
                    .collect()
            } else {
                (0..len)
                    .map(|index| {
                        (index + 1..=index + k)
                            .map(|i| code[i.rem_euclid(len) as usize])
                            .sum()
                    })
                    .collect()
            }
        }
    }
}

// <=======================================================================>

macro_rules! c_ {
    ($l: ident, $r: ident, $i: ident) => {
        ($l.is_none() || $l.unwrap() + 1 < $i) && ($r.is_none() || $r.unwrap() - 1 > $i)
    };
}

pub fn find_lonely(mut n: Vec<i32>) -> Vec<i32> {
    n.sort_unstable();
    n.iter().enumerate().fold(Vec::new(), |mut ret, (idx, &i)| {
        let l = n.get(idx - 1);
        let r = n.get(idx + 1);
        if c_!(l, r, i) {
            ret.push(i);
        }
        ret
    })
}

// <=======================================================================>

macro_rules! c {
    ($m: ident, $i: expr) => {
        $m[$i].eq(&1) && $m[$i - 1].eq(&0) && $m[$i + 1].eq(&0)
    };
}

macro_rules! p {
    ($m: ident, $n: ident) => {
        $n.iter().for_each(|&i| $m[i as usize + 1] += 1)
    };
}

pub fn find_lonely_(n: Vec<i32>) -> Vec<i32> {
    let mut m = [0; 7 + 10usize.pow(6)];
    p!(m, n);
    n.iter().fold(Vec::new(), |mut ret, &i| {
        let i_ = i as usize + 1;
        if c!(m, i_) {
            ret.push(i);
        }
        ret
    })
}

// <=======================================================================>

macro_rules! c__ {
    ($ret: expr) => {{
        let r = $ret;
        if r.eq(&i32::MAX) {
            -1
        } else {
            r
        }
    }};
}

pub fn minimum_subarray_length(n: Vec<i32>, k: i32) -> i32 {
    c__!(
        n.iter()
            .enumerate()
            .fold(
                (0, 0, i32::MAX, [0; 32]),
                |(mut i, mut t, mut ret, mut bits), (j, nj)| {
                    t |= nj;
                    (0..32).rev().for_each(|b| bits[b] += (nj >> b) & 1);
                    while i <= j && t >= k {
                        ret = ret.min((j - i + 1) as i32);
                        (0..32).rev().for_each(|b| {
                            bits[b] -= (n[i] >> b) & 1;
                            if bits[b].eq(&0) {
                                t &= !(1 << b);
                            }
                        });
                        i += 1;
                    }
                    (i, t, ret, bits)
                }
            )
            .2
    )
}

// <=======================================================================>

pub fn sum_of_the_digits_of_harshad_number(x: i32) -> i32 {
    fn digits(x: &i32) -> i32 {
        if *x < 10 {
            *x
        } else {
            x % 10 + digits(&(x / 10))
        }
    }
    let s = digits(&x);
    if (x % s).eq(&0) {
        s
    } else {
        -1
    }
}

// <=======================================================================>

macro_rules! bin {
    ($n: ident, $ko: expr, $t: ident, $k: ident, $i: ident, $ni: expr) => {
        match $n[$ko..].binary_search_by_key(&($t - *$ni), |&(_, nj)| *nj) {
            Ok(ok) => Some(vec![*$i as i32, $n[ok + $k + 1].0 as i32]),
            Err(_) => None,
        }
    };
}

pub fn two_sum(n: Vec<i32>, t: i32) -> Vec<i32> {
    let mut n = n.iter().enumerate().collect::<Vec<_>>();
    n.sort_unstable_by_key(|&(_, x)| x);
    n.iter()
        .enumerate()
        .filter_map(|(k, (i, ni))| bin!(n, k + 1, t, k, i, ni))
        .next()
        .unwrap_or(Vec::new())
}

// <=======================================================================>

pub fn level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
    fn traverse(
        root: &Option<&Rc<RefCell<TreeNode>>>,
        ret: &mut Vec<Vec<i32>>,
        n: &mut usize,
        lvl: &usize,
    ) {
        if let Some(node) = root {
            if n.eq(&lvl) {
                ret.push(Vec::new());
                *n += 1;
            }
            ret[*lvl].push(node.borrow().val);
            traverse(&node.borrow().left.as_ref(), ret, n, &(lvl + 1));
            traverse(&node.borrow().right.as_ref(), ret, n, &(lvl + 1));
        }
    }
    let mut ret = Vec::new();
    traverse(&root.as_ref(), &mut ret, &mut 0usize, &0);
    ret
}

// <=======================================================================>

macro_rules! gm {
    ($ws: ident, $n: ident) => {
        $ws.iter()
            .map(|w| {
                $n += 1;
                w.as_bytes().iter().fold((0, 0), |(acc, len), x| {
                    (acc | 1 << (x - 'a' as u8), len + 1)
                })
            })
            .collect::<Vec<_>>()
    };
}

pub fn max_product(ws: Vec<String>) -> i32 {
    let (mut n, mut m) = (0, 0);
    let mp = gm!(ws, n);

    for i in 0..n - 1 {
        for j in i + 1..n {
            if (mp[i].0 & mp[j].0).eq(&0) {
                m = m.max(mp[i].1 * mp[j].1)
            }
        }
    }
    m
}

// <=======================================================================>

pub fn longest_monotonic_subarray(nums: Vec<i32>) -> i32 {
    nums.windows(2)
        .fold((1, 1, 1), |(inc, dec, max), w| {
            sv(&w[1], &w[0], &inc, &dec, &max)
        })
        .2
}

fn sv(ni: &i32, nj: &i32, inc: &i32, dec: &i32, max: &i32) -> (i32, i32, i32) {
    if ni > nj {
        ((inc + 1), 1, *max.max(&(inc + 1)).max(&1))
    } else if ni < nj {
        (1, dec + 1, *max.max(&1).max(&(dec + 1)))
    } else {
        (1, 1, *max.max(&1))
    }
}

// <=======================================================================>

pub fn longest_monotonic_subarray_basic(nums: Vec<i32>) -> i32 {
    nums.windows(2)
        .fold((1, 1, 1), |(inc, dec, max), w| {
            if w[1] > w[0] {
                ((inc + 1), 1, max.max(inc + 1).max(1))
            } else if w[1] < w[0] {
                (1, dec + 1, max.max(dec + 1).max(1))
            } else {
                (1, 1, max.max(1))
            }
        })
        .2
}

// <=======================================================================>

type TL = Rc<RefCell<TreeNode>>;

pub fn kth_smallest(root: Option<TL>, k: i32) -> i32 {
    (0..=0)
        .fold((Vec::new(), 0xFFFFFFF), |(mut mins, _), _| {
            fn traverse__(root: &Option<&TL>, k: &i32, mins: &mut Vec<i32>) {
                if let Some(node) = root {
                    mins.push(node.borrow().val);
                    traverse__(&node.borrow().left.as_ref(), &k, mins);
                    traverse__(&node.borrow().right.as_ref(), &k, mins);
                }
            }
            traverse__(&root.as_ref(), &k, &mut mins);
            mins.sort_unstable();
            let kmin = mins[k as usize - 1];
            (mins, kmin)
        })
        .1
}

// <=======================================================================>

#[allow(unused)]
struct Bank {
    bals: Vec<i64>,
    size: i32,
}

#[allow(unused)]
impl Bank {
    #[inline]
    fn new(bals: Vec<i64>) -> Bank {
        let size = bals.len() as i32;
        Bank { bals, size }
    }

    #[inline]
    fn check_bounds(&self, acc1: &i32, acc2: &i32) -> bool {
        *acc1 <= self.size && *acc2 <= self.size
    }

    #[inline]
    fn get_bal(&mut self, acc: &i32) -> i64 {
        self.bals[*acc as usize - 1]
    }

    #[inline]
    fn get_mut_bal(&mut self, acc: &i32) -> &mut i64 {
        &mut self.bals[*acc as usize - 1]
    }

    fn transfer(&mut self, acc1: i32, acc2: i32, money: i64) -> bool {
        if self.check_bounds(&acc1, &acc2) && self.get_bal(&acc1) >= money {
            *self.get_mut_bal(&acc1) -= money;
            *self.get_mut_bal(&acc2) += money;
            true
        } else {
            false
        }
    }

    fn deposit(&mut self, acc: i32, money: i64) -> bool {
        if self.check_bounds(&acc, &0) {
            *self.get_mut_bal(&acc) += money;
            true
        } else {
            false
        }
    }

    fn withdraw(&mut self, acc: i32, money: i64) -> bool {
        if self.check_bounds(&acc, &0) && self.get_bal(&acc) >= money {
            *self.get_mut_bal(&acc) -= money;
            true
        } else {
            false
        }
    }
}

// <=======================================================================>

pub fn score_of_string(s: String) -> i32 {
    s.chars()
        .collect::<Vec<_>>()
        .windows(2)
        .map(|w| (w[1] as i32 - w[0] as i32).abs())
        .sum::<i32>()
}

// <=======================================================================>

pub fn min_rectangles_to_cover_points(mut pts: Vec<Vec<i32>>, w: i32) -> i32 {
    (0..=0).fold(pts.len(), |n, _| {
        pts.sort_unstable();
        (0..n)
            .fold((1, w + pts[0][0]), |(ret, next), i| {
                if pts[i][0] > next {
                    (ret + 1, pts[i][0] + w)
                } else {
                    (ret, next)
                }
            })
            .0
    }) as i32
}

// <=======================================================================>

fn mod_exp(mut base: u64, mut exp: u64, modu: u64) -> u64 {
    let mut ret = 1;
    base %= modu;
    while exp > 0 {
        if (exp & 1).eq(&1) {
            ret = (ret * base) % modu;
        }
        exp >>= 1;
        base = (base * base) % modu;
    }
    ret
}

// https://en.wikipedia.org/wiki/Xorshift
fn prng(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

const ITERATIONS: u64 = 4;
const XORSHIFT_STATE: u64 = 0xFFFFFFFFFFFFFFFF;

// https://en.wikipedia.org/wiki/Fermat_primality_test
fn fermat(n: u64, mut state: u64) -> bool {
    if n <= 1 {
        return false;
    } else if n <= 3 {
        return true;
    }
    for _ in 0..ITERATIONS {
        let a = prng(&mut state) % (n - 1) + 1;
        if mod_exp(a, n - 1, n) != 1 {
            return false;
        }
    }
    true
}

pub fn maximum_prime_difference(nums: Vec<i32>) -> i32 {
    ((0..nums.len())
        .find(|&i| fermat(nums[i] as u64, XORSHIFT_STATE))
        .unwrap() as i32)
        .abs_diff(
            (0..nums.len())
                .rev()
                .find(|&i| fermat(nums[i] as u64, XORSHIFT_STATE))
                .unwrap() as i32,
        ) as i32
}

// <=======================================================================>

const PRIMES: [bool; 101] = [
    false, false, true, true, false, true, false, true, false, false, false, true, false, true,
    false, false, false, true, false, true, false, false, false, true, false, false, false, false,
    false, true, false, true, false, false, false, false, false, true, false, false, false, true,
    false, true, false, false, false, true, false, false, false, false, false, true, false, false,
    false, false, false, true, false, true, false, false, false, false, false, true, false, false,
    false, true, false, true, false, false, false, false, false, true, false, false, false, true,
    false, false, false, false, false, true, false, false, false, false, false, false, false, true,
    false, false, false,
];

pub fn maximum_prime_difference_(nums: Vec<i32>) -> i32 {
    ((0..nums.len()).find(|&i| PRIMES[nums[i] as usize]).unwrap() as i32).abs_diff(
        (0..nums.len())
            .rev()
            .find(|&i| PRIMES[nums[i] as usize])
            .unwrap() as i32,
    ) as i32
}

// <=======================================================================>

pub fn find_latest_time(s: String) -> String {
    s.chars()
        .enumerate()
        .fold(String::new(), |mut ret, (i, c)| {
            ret.push(if !c.eq(&'?') {
                c
            } else {
                match i {
                    0 => {
                        let next = s.chars().nth(1).unwrap();
                        if next.eq(&'1') || next.eq(&'?') || next.eq(&'0') {
                            '1'
                        } else {
                            '0'
                        }
                    }
                    1 => {
                        let prev = s.chars().nth(0).unwrap();
                        if prev.eq(&'1') || prev.eq(&'?') {
                            '1'
                        } else {
                            '9'
                        }
                    }
                    3 => '5',
                    _ => '9',
                }
            });
            ret
        })
}

// <=======================================================================>

pub fn count_even(num: i32) -> i32 {
    (2..=num)
        .filter(|i| {
            let (mut sum, mut x) = (0, *i);
            while x > 0 {
                sum += x % 10;
                x /= 10;
            }
            (sum & 1).eq(&0)
        })
        .count() as i32
}

// <=======================================================================>

pub fn find_lucky(arr: Vec<i32>) -> i32 {
    (0..=0).fold(-0xFFFFFFF, |_, _| {
        let map = arr.iter().fold([0; 501], |mut map, i| {
            map[*i as usize] += 1;
            map
        });
        arr.iter().fold(-1, |ret, i| {
            if i.eq(&map[*i as usize]) {
                ret.max(*i)
            } else {
                ret
            }
        })
    })
}

// <=======================================================================>

pub fn pick_gifts(gifts: Vec<i32>, mut k: i32) -> i64 {
    (0..=0).fold(-0xFFFFFFFFFFFFFF, |_, _| {
        if gifts.len().eq(&1) {
            return 1;
        } else if gifts[0].eq(&gifts[gifts.len() - 1]) {
            return gifts.iter().sum::<i32>() as i64;
        }

        let mut pq = std::collections::BinaryHeap::<i64>::from(
            gifts.iter().map(|x| *x as i64).collect::<Vec<_>>(),
        );
        while k > 0 {
            let i = pq.pop().unwrap();
            pq.push((i as f64 + 0.5).sqrt() as i64);
            k -= 1;
        }
        pq.iter().sum()
    })
}

// <=======================================================================>

const MONTHS: &[&str] = &[
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

pub fn reformat_date(date: String) -> String {
    (0..=0).fold(String::new(), |_, _| {
        let s = date.split_whitespace().collect::<Vec<_>>();
        format!(
            "{}-{:02}-{:02}",
            s[2],
            MONTHS.iter().position(|m_| m_.eq(&s[1])).unwrap() + 1,
            s[0][..s[0].len() - 2].parse::<u8>().unwrap()
        )
    })
}

// <=======================================================================>

pub fn count_asterisks(s: String) -> i32 {
    s.split('|')
        .step_by(2)
        .map(|g| g.chars().filter(|c| c.eq(&'*')).count())
        .sum::<usize>() as i32
}

// <=======================================================================>

pub fn check_x_matrix(grid: Vec<Vec<i32>>) -> bool {
    let n = grid.len();
    for y in 0..n {
        for x in 0..n {
            if ((y.eq(&x) || ((y + x).eq(&(n - 1)))) && grid[y][x].eq(&0))
                || !(y.eq(&x) || ((y + x).eq(&(n - 1)))) && !grid[y][x].eq(&0)
            {
                return false;
            }
        }
    }
    true
}

// <=======================================================================>

pub fn number_of_special_chars(word: String) -> i32 {
    word.chars()
        .fold([0; 26], |mut map, c| {
            let i = c.to_ascii_lowercase() as usize - 97; // 'a'
            if c.is_uppercase() {
                map[i] |= 0b10;
            } else {
                map[i] |= 0b01;
            }
            map
        })
        .iter()
        .filter(|x| x.eq(&&0b11))
        .count() as i32
}

// <=======================================================================>

pub fn find_kth_positive(arr: Vec<i32>, mut k: i32) -> i32 {
    arr.iter().any(|&x| {
        if x <= k {
            k += 1
        }
        x >= k
    });
    k
}

// <=======================================================================>

pub fn k_length_apart_(nums: Vec<i32>, k: i32) -> bool {
    for w in nums
        .iter()
        .enumerate()
        .filter_map(|(i, x)| if x.eq(&1) { Some(i) } else { None })
        .collect::<Vec<_>>()
        .windows(2)
    {
        if w[1] - w[0] <= k as usize {
            return false;
        }
    }
    true
}

// <=======================================================================>

pub fn k_length_apart(nums: Vec<i32>, k: i32) -> bool {
    nums.iter()
        .enumerate()
        .fold((-0xFFFFFFF, true), |(mut l, mut r), (i, x)| {
            if x.eq(&1) {
                if !l.eq(&-0xFFFFFFF) {
                    if i as i32 - l - 1 < k {
                        r = false
                    }
                }
                l = i as i32;
            }
            (l, r)
        })
        .1
}

// <=======================================================================>

pub fn maximum_difference(nums: Vec<i32>) -> i32 {
    let n = nums.len();
    let mut ret = -1;
    for i in 0..n - 1 {
        for j in i + 1..n {
            if nums[i] < nums[j] {
                ret = ret.max(nums[j] - nums[i]);
            }
        }
    }
    ret
}

// <=======================================================================>

pub fn most_frequent_even(nums: Vec<i32>) -> i32 {
    let mut evens = nums.iter().filter(|i| *i & 1 == 0).peekable();
    if evens.peek().is_none() {
        return -1;
    }

    let (map, max) =
        evens
            .into_iter()
            .fold((HashMap::new(), -0xFFFFFFF), |(mut map, mut max), i| {
                let k = map.entry(i).or_insert(0);
                *k += 1;
                if *k > max {
                    max = *k;
                }
                (map, max)
            });

    **map
        .iter()
        .filter(|(_, v)| **v == max)
        .map(|(k, _)| k)
        .min()
        .unwrap()
}

// <=======================================================================>

const POSITIONS: &[&[(usize, usize); 4]] = &[
    &[(0, 0), (0, 1), (1, 0), (1, 1)],
    &[(0, 1), (0, 2), (1, 1), (1, 2)],
    &[(1, 0), (1, 1), (2, 0), (2, 1)],
    &[(1, 1), (1, 2), (2, 1), (2, 2)],
];

pub fn can_make_square(grid: Vec<Vec<char>>) -> bool {
    POSITIONS.iter().any(|&p| {
        match p.iter().fold((-2, -2), |(w, b), p| {
            if grid[p.0][p.1].eq(&'W') {
                (w + 1, b)
            } else {
                (w, b + 1)
            }
        }) {
            (a, b) => [a, b],
        }
        .iter()
        .any(|x| *x > 0)
    })
}

// <=======================================================================>

pub fn added_integer(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
    nums2.iter().min().unwrap() - nums1.iter().min().unwrap()
}

// <=======================================================================>

pub fn find_length_of_lcis(nums: Vec<i32>) -> i32 {
    *match nums.windows(2).fold((1, 1), |(max, curr), w| {
        if w[1] > w[0] {
            (max, curr + 1)
        } else {
            (max.max(curr), 1)
        }
    }) {
        (a, b) => vec![a, b],
    }
    .iter()
    .max()
    .unwrap()
}

// <=======================================================================>

const VS: &[char; 5] = &['a', 'e', 'i', 'o', 'u'];

pub fn is_valid(word: String) -> bool {
    (0..=0)
        .fold(0b1000011, |mut ret, _| {
            if word.len() < 3 {
                return ret;
            }
            if word.as_bytes().iter().any(|b| {
                if VS.contains(&(*b as char).to_ascii_lowercase()) {
                    ret &= !(1 << 1);
                    false
                } else if b.is_ascii_alphabetic() {
                    ret |= 1 << 2;
                    false
                } else {
                    !b.is_ascii_alphanumeric()
                }
            }) {
                0
            } else {
                ret
            }
        })
        .eq(&69)
}

// <=======================================================================>

macro_rules! a {
    (.|.$c: expr) => {
        *$c as usize - 0x61
    };
    (|..|$it: expr) => {
        $it.into_iter().enumerate()
    };
}

pub fn find_permutation_difference(s: String, t: String) -> i32 {
    (0..=0)
        .fold((0x45, [0x0; 0x1A]), |(_, mut m), _| {
            a!(|..| t.as_bytes()).for_each(|(i, c)| m[a!(.|.c)] = i);
            (
                a!(|..| s.as_bytes()).fold(0x0, |r, (i, c)| r + m[a!(.|.c)].abs_diff(i)),
                m,
            )
        })
        .0 as i32
}

// <=======================================================================>

pub fn is_array_special(nums: Vec<i32>) -> bool {
    !(0..=0)
        .fold((false, nums.windows(2)), |(r, mut i), _| {
            if nums.len() <= 1 {
                return (r, i);
            }
            (i.any(|w| w[0] & 1 ^ w[1] & 1 == 0), i)
        })
        .0
}

// <=======================================================================>

pub fn sum_digit_differences(mut nums: Vec<i32>) -> i64 {
    nums.iter_mut()
        .fold([[0; 10]; 10], |mut f, x| {
            (0..10).any(|i| {
                f[i][*x as usize % 10] += 1;
                *x /= 10;
                *x == 0
            });
            f
        })
        .iter()
        .fold((0, nums.len()), |(mut r, n), f| {
            f.iter().for_each(|x| r += (n - x) * x);
            (r, n)
        })
        .0 as i64
        >> 1
}

// <=======================================================================>

pub fn subset_xor_sum(nums: Vec<i32>) -> i32 {
    nums.iter().fold(0x0, |ret, x| ret | x) << nums.len() - 1
}

// <=======================================================================>

// codewars
pub fn is_pangram(s: &str) -> bool {
    s.chars()
        .filter(|x| x.is_alphabetic())
        .map(|x| x.to_ascii_lowercase())
        .collect::<std::collections::HashSet<_>>()
        .len()
        .eq(&26)
}

// <=======================================================================>

// codewars
pub fn remove_every_other(nums: &[u8]) -> Vec<u8> {
    nums.iter()
        .enumerate()
        .filter_map(|(i, x)| if i & 1 == 0 { Some(*x) } else { None })
        .collect()
}

// <=======================================================================>

// codewars
pub fn order(s: &str) -> String {
    let mut cs = s
        .split_whitespace()
        .fold(Vec::with_capacity(100), |mut ret, x| {
            let dx = x
                .chars()
                .find(|x| x.is_digit(10))
                .unwrap()
                .to_digit(10)
                .unwrap();
            ret.push((dx, x));
            ret
        });
    cs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    cs.iter().map(|(_, s)| *s).collect::<Vec<_>>().join(" ")
}

// <=======================================================================>

// codewars
pub fn cakes(r: &HashMap<&str, u32>, a: &HashMap<&str, u32>) -> u32 {
    r.into_iter()
        .fold(!0, |r, (k, v)| r.min(a.get(k).unwrap_or(&0) / v))
}

// <=======================================================================>

// codewars
pub fn sequence_sum(s: u32, e: u32, step: u32) -> u32 {
    (s..=e).step_by(step as usize).sum::<u32>()
}

// <=======================================================================>

pub fn subtract_product_and_sum(mut n: i32) -> i32 {
    let (mut sum, mut prod) = (0, 1);
    while n > 0 {
        prod *= n % 10;
        sum += n % 10;
        n /= 10;
    }
    prod - sum
}

// <=======================================================================>

pub fn duplicate_numbers_xor(nums: Vec<i32>) -> i32 {
    nums.iter()
        .fold(HashMap::with_capacity(50), |mut map, x| {
            *map.entry(x).or_insert(0) += 1;
            map
        })
        .iter()
        .filter(|(_, x)| x.eq(&&2))
        .fold(0x0, |ret, (x, _)| ret ^ *x)
}

// <=======================================================================>

pub fn occurrences_of_element(nums: Vec<i32>, queries: Vec<i32>, x: i32) -> Vec<i32> {
    (0..=0).fold(
        nums.iter()
            .enumerate()
            .filter(|(_, n)| **n == x)
            .map(|(i, _)| i as i32)
            .collect::<Vec<_>>(),
        |occs, _| {
            queries
                .iter()
                .map(|q| {
                    if *q as usize <= occs.len() {
                        occs[*q as usize - 1]
                    } else {
                        -1
                    }
                })
                .collect::<Vec<_>>()
        },
    )
}

// <=======================================================================>

pub fn query_results(_: i32, qs: Vec<Vec<i32>>) -> Vec<i32> {
    qs.iter()
        .fold(
            (
                HashMap::<i32, i32>::new(),
                HashMap::<i32, i32>::new(),
                Vec::new(),
                0,
            ),
            |(mut bcs, mut ccs, mut ret, mut dcc), q| {
                if let Some(c) = ccs.get_mut(bcs.get(&q[0]).unwrap_or(&-0xAAA)) {
                    *c -= 1;
                    if *c == 0 {
                        dcc -= 1;
                    }
                }
                bcs.insert(q[0], q[1]);
                let c = ccs.entry(q[1]).or_insert(0);
                *c += 1;
                if *c == 1 {
                    dcc += 1;
                }
                ret.push(dcc);
                (bcs, ccs, ret, dcc)
            },
        )
        .2
}

// <=======================================================================>

pub fn compressed_string(word: String) -> String {
    (0..=0).fold(
        "upvote that and your mom will live forever".to_owned(),
        |_, _| {
            let bytes = word.as_bytes();
            let (mut ret, j) = bytes.into_iter().enumerate().skip(1).fold(
                (Vec::new(), 0),
                |(mut ret, j), (i, xi)| {
                    if *xi != bytes[j] || i > j + 8 {
                        ret.extend([48 + (i - j) as u8, bytes[j]]);
                        (ret, i)
                    } else {
                        (ret, j)
                    }
                },
            );
            ret.extend([48 + (bytes.len() - j) as u8, bytes[j]]);
            unsafe { String::from_utf8_unchecked(ret) }
        },
    )
}

// <=======================================================================>

use std::ops::Add;

pub fn find_indices(nums: Vec<i32>, id: i32, vd: i32) -> Vec<i32> {
    if id.add(vd).eq(&0) {
        return vec![0, 0];
    }
    let n = nums.len();
    for i in 0..n {
        for j in 0..n {
            if i.abs_diff(j) >= id as _ && nums[i].abs_diff(nums[j]) >= vd as _ {
                return vec![i as i32, j as i32];
            }
        }
    }
    vec![-1, -1]
}

// <=======================================================================>

pub fn replace_words_(mut d: Vec<String>, s: String) -> String {
    s.split_whitespace()
        .fold((Vec::new(), d.sort_unstable()), |(mut ret, _), w| {
            ret.push(
                d.iter()
                    .find(|r| w.starts_with(*r))
                    .map(String::as_str)
                    .unwrap_or(w),
            );
            (ret, ())
        })
        .0
        .join(" ")
}

pub fn replace_words(mut d: Vec<String>, s: String) -> String {
    s.split_whitespace()
        .fold(
            (
                {
                    d.sort_unstable();
                    d.dedup_by(|r, p| r.starts_with(p.as_str()))
                },
                Vec::with_capacity(s.len()),
            ),
            |(_, mut ret), w| {
                ret.push(match d.partition_point(|d| d.as_str() < w) {
                    i @ (1..) if w.starts_with(&d[i - 1]) => d[i - 1].as_str(),
                    _ => w,
                });
                ((), ret)
            },
        )
        .1
        .join(" ")
}

// <=======================================================================>

pub fn clear_digits(s: String) -> String {
    s.chars().fold(String::with_capacity(s.len()), |mut s, c| {
        if c.is_ascii_digit() {
            s.pop();
        } else {
            s.push(c);
        }
        s
    })
}

// <=======================================================================>

pub fn find_winning_player(s: Vec<i32>, k: i32) -> i32 {
    (1..s.len())
        .try_fold((0, 0), |(c, m), i| {
            if c >= k {
                Err((c, m))
            } else if s[i] < s[m] {
                Ok((c + 1, m))
            } else {
                Ok((1, i))
            }
        })
        .unwrap_or_else(|(c, m)| (c, m))
        .1 as i32
}

// <=======================================================================>

pub fn count_complete_day_pairs(hs: Vec<i32>) -> i32 {
    let mut ret = 0;
    let n = hs.len();
    for i in 0..n - 1 {
        for j in i..n {
            if i < j && (hs[i] + hs[j]) % 24 == 0 {
                ret += 1;
            }
        }
    }
    ret
}

// <=======================================================================>

pub fn twos_difference(v: &[u32]) -> Vec<(u32, u32)> {
    use std::collections::HashSet;

    let set = v.iter().collect::<HashSet<_>>();

    let mut ret = v
        .iter()
        .filter_map(|i| {
            if set.contains(&(i + 2)) {
                Some((*i, i + 2))
            } else {
                None
            }
        }).collect::<Vec<_>>();

    ret.sort_unstable();
    ret
}

pub fn get_encrypted_string(s: String, k: i32) -> String {
    let bytes = s.as_bytes();
    s.char_indices().map(|(i, _)| {
        bytes[(i + k as usize) % bytes.len()] as char
    }).collect()
}

pub fn final_position_of_snake(n: i32, commands: Vec::<String>) -> i32 {
    let mut dir = (0, 0);
    commands.into_iter().for_each(|cmd| {
        match cmd.as_str() {
            "UP"    => dir.0 -= 1,
            "DOWN"  => dir.0 += 1,
            "LEFT"  => dir.1 -= 1,
            "RIGHT" => dir.1 += 1,
            _       => unreachable!()
        };
    });

    (dir.0 * n) + dir.1
}

pub fn find_different(nums: &Vec::<i32>, common: i32) -> i32 {
    nums.iter().sum::<i32>() - (nums.len() as i32 * common) + common
}

#[allow(unused)]
macro_rules! tovsstring {
    ($($str: expr), *) => { vec![$($str.to_owned()), *] }
}

#[allow(unused)]
macro_rules! own {
    ($str: expr) => {
        $str.to_owned()
    };
}

#[allow(unused)]
macro_rules! map {
    () => { HashMap::new() };
    ($($i: ident: $a: expr), +) => {{
        let mut map = HashMap::new();
        $(map.insert(stringify!($i), $a);)*
        map
    }};
}

pub fn is_balanced(s: String) -> bool {
    let (e, o) = s.as_bytes().iter().enumerate().fold((0, 0), |(even, odd), (i, b)| {
        if i & 1 == 0 {
            (even + b - b'0', odd)
        } else {
            (even, odd + b - b'0')
        }
    }); e == o
}

pub fn smallest_number(n: i32, t: i32) -> i32 {
    fn dig_prod(mut x: i32) -> i32 {
        let mut prod = 1;
        while x != 0 {
            prod *= x % 10;
            x /= 10;
        } prod
    }

    let mut curr = n;
    loop {
        if dig_prod(curr) % t == 0 {
            return curr
        } curr += 1;
    }
}

pub fn find_subtree_sizes(parents: Vec::<i32>, s: String) -> Vec::<i32> {
    fn dfs(
        mut parent: i32,
        node: i32,

        ret: &mut Vec::<i32>,
        letter_stack: &mut Vec::<Vec::<i32>>,
        children: &Vec::<Vec::<i32>>,

        s: &[u8]
    ) {
        let unode = node as usize;
        let ch = (s[unode] - const { b'a' }) as usize;
        if let Some(&child) = letter_stack[ch].last() {
            parent = child
        }
        letter_stack[ch].push(node);
        children[unode].iter().for_each(|&next| {
            dfs(node, next, ret, letter_stack, children, s)
        });
        letter_stack[ch].pop();
        ret[unode] += 1;
        if parent != -1 {
            ret[parent as usize] += ret[unode]
        }
    }

    let n = parents.len();
    let mut children = vec![Vec::new(); n];
    (1..n).for_each(|i| {
        children[parents[i] as usize].push(i as i32)
    });

    let mut ret = vec![0; n];
    dfs(-1, 0, &mut ret, &mut vec![Vec::new(); 26], &children, s.as_bytes());
    ret
}

pub fn possible_string_count(word: String) -> i32 {
    let mut bytes = word.as_bytes().to_vec();
    bytes.dedup();
    (word.len() - bytes.len()) as i32 + 1
}

pub fn min_element(nums: Vec::<i32>) -> i32 {
    fn dig_sum(mut x: i32) -> i32 {
        let mut sum = 0;
        while x != 0 {
            sum += x % 10;
            x /= 10;
        } sum
    }

    unsafe { nums.into_iter().map(dig_sum).min().unwrap_unchecked() }
}

// pub fn report_spam(msg: Vec::<String>, banned: Vec::<String>) -> bool {
//     (0x45..=69).fold((false, banned.iter().collect::<std::collections::HashSet::<_>>()), |(.., set), _| {
//         let Some(pos) = msg.iter().position(|w| banned.contains(w)) else { return (false, set) };
//         (msg.iter().skip(pos + 1).any(|w| banned.contains(w)), set)
//     }).0
// }

pub fn report_spam(msg: Vec::<String>, banned: Vec::<String>) -> bool {
    let banned = banned.into_iter().collect::<std::collections::HashSet::<_>>();
    let Some(pos) = msg.iter().position(|w| banned.contains(w)) else { return false };
    msg.iter().skip(pos + 1).any(|w| banned.contains(w))
}

pub fn count_good_nodes(edges: Vec::<Vec::<i32>>) -> i32 {
    fn dfs(node: i32, parent: i32, adjs: &[Vec::<i32>], good_nodes_count: &mut i32) -> i32 {
        let (mut subtree_size, mut child_size, mut ok) = (1, 0, true);
        for &neighbor in adjs[node as usize].iter() {
            if neighbor == parent { continue }
            let child_subtree_size = dfs(neighbor, node, adjs, good_nodes_count);
            if child_size == 0 {
                child_size = child_subtree_size
            } else if child_size != child_subtree_size {
                ok = false
            } subtree_size += child_subtree_size
        }

        if ok {
            *good_nodes_count += 1
        } subtree_size
    }

    let mut adjs = vec![Vec::new(); edges.len() + 1];
    for e in edges {
        let (a, b) = (e[0], e[1]);
        adjs[a as usize].push(b);
        adjs[b as usize].push(a);
    }

    let mut ret = 0;
    dfs(0, -1, &adjs, &mut ret);
    ret
}

fn main() {
    dbg!(count_good_nodes(vec![vec![0,1],vec![0,2],vec![1,3],vec![1,4],vec![2,5],vec![2,6]]));
    dbg!(count_good_nodes(vec![vec![0, 1], vec![1, 2], vec![1, 3], vec![1, 4], vec![0, 5], vec![5, 6], vec![6, 7], vec![7, 8], vec![0, 9], vec![9, 10], vec![9, 12], vec![10, 11]]));
    dbg!(report_spam(tovsstring!["hello","world","leetcode"], tovsstring!["world","hello"]));
    dbg!(report_spam(tovsstring!["hello","programming","fun"], tovsstring!["world","programming","leetcode"]));
    dbg!(min_element(vec![10,12,13,14]));
    dbg!(possible_string_count("abbcccc".to_owned()));
    dbg!(possible_string_count("abcd".to_owned()));
    dbg!(possible_string_count("aaaa".to_owned()));
    dbg!(find_subtree_sizes(vec![-1,0,0,1,1,1], "abaabc".to_owned()));
    dbg!(smallest_number(15, 3));
    dbg!(is_balanced("24123".to_owned()));
    dbg!(find_different(&vec![5, 5, 5, 7, 5, 5], 5));
    dbg!(final_position_of_snake(2, tovsstring!("RIGHT","DOWN")));
    dbg!(get_encrypted_string("dart".to_owned(), 3));
    dbg!(count_complete_day_pairs(vec![72, 48, 24, 3]));
    dbg!(find_winning_player(vec![4, 2, 6, 3, 9], 2));
    // dbg!(clear_digits(own!("cb34")));
    // dbg!(replace_words(vec![own!("cat"), own!("bat"), own!("rat")], own!("the cattle was rattled by the battery")));
    // dbg!(find_indices(vec![5,1,4,1], 2, 4));
    // dbg!(compressed_string("aaaaaaaaaaaaaabb".to_owned()));
    // dbg!(subtract_product_and_sum(234));
    // dbg!(sequence_sum(2, 6, 2));
    // dbg!(order("is2 Thi1s T4est 3a"));
    // dbg!(remove_every_other(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
    // dbg!(is_pangram("The quick, brown fox jumps over the lazy dog!"));
    // dbg!(sum_digit_differences(vec![13,23,12]));
    // dbg!(is_array_special(vec![2,1,4]));
    // dbg!(find_permutation_difference(own!("abc"), own!("bac")));
    // dbg!(is_valid(own!("AhI")));
    // dbg!(find_length_of_lcis(vec![2,2,2,2,2]));
    // dbg!(can_make_square(vec![vec!['B','W','B'], vec!['W','B','W'], vec!['B','W','B']]));
    // dbg!(most_frequent_even(vec![0,1,2,2,4,4,1]));
    // dbg!(maximum_difference(vec![1,5,2,10]));
    // dbg!(k_length_apart(vec![1,0,0,0,1,0,0,1], 2));
    // dbg!(find_kth_positive(vec![2, 3, 4, 7, 11], 5));
    // dbg!(number_of_special_chars(own!("aaAbcBC")));
}
