    //                               l        r       u        d
pub fn num_islands(mut grid: Vec<Vec<char>>) -> i32 {
    const DIRS: [(i32, i32); 4] = [(0, -1), (0, 1), (1, 0), (-1, 0)];
    fn dfs(i: i32, j: i32, grid: &mut Vec<Vec<char>>, n: &i32, n0: &i32) -> bool {
        if i < 0
        || j < 0
        || i >= *n
        || j >= *n0
        || grid[i as usize][j as usize] == '0'
        {
            return false;
        }
        grid[i as usize][j as usize] = '0';
        for (di, dj) in DIRS {
            dfs(i + di, j + dj, grid, n, n0);
        } true
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
    } ans
}

// <=======================================================================>

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
            ListNode {
            next: None,
            val
        }
    }
}

fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        if b < a { std::mem::swap(&mut b, &mut a); } 
        b %= a; 
    } a
}

pub fn insert_greatest_common_divisors(mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut current = &mut head;

    while let Some(node) = current {
        if let Some(next_node) = node.next.take() {
            let mut new_node = Box::new(ListNode::new(gcd(node.val, next_node.val)));
            new_node.next = Some(next_node);

            node.next = Some(new_node);
            current = &mut node.next.as_mut().unwrap().next;
        } else { break; }
    }
    head
}

// <=======================================================================>

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

// <=============================================================================>

#[allow(unused)]
struct RandomizedSet {
    n: usize,
    state: usize,
    map: HashMap<i32, usize>,
    values: [Option<i32>; 5000]
}

#[allow(unused)]
impl RandomizedSet {
    fn new() -> RandomizedSet  {
        RandomizedSet {
            n: 0,
            state: 1488,
            map: HashMap::with_capacity(200000),
            values: [None; 5000]
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
        } else { false }
    }

    fn get_random(&mut self) -> i32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.values[self.state %  self.n].unwrap()
    }
}


// <=============================================================================>

use std::collections::HashSet;

struct RandomizedCollection {
    map: HashMap<i32, (HashSet<usize>, usize)>,
    values: [Option<i32>; 10000],
    state: usize,
    n: usize
}

#[allow(unused)]
impl RandomizedCollection {
    fn new() -> RandomizedCollection {
        RandomizedCollection {
            map: HashMap::new(),
            values: [None; 10000],
            state: 1488,
            n: 0
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
            let i    = *set.iter().next().unwrap();
            let end  = self.n - 1;

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
            } true
        } else { false }
    }

    fn get_random(&mut self) -> i32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.values[self.state %  self.n].unwrap()
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
    } mat
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
    words.iter().fold((0, HashMap::<&str, (i64, usize)>::new()), |(mut acc, mut map), w| {
        let n = w.len();
        map.iter().for_each(|(k, (v, len))| {
            if *len <= n
                && w.starts_with(k)
                && w.ends_with(k) {
                    acc += v;
                }
        });
        map.entry(w.as_str()).or_insert_with(|| (0, n)).0 += 1;
        (acc, map)
    }).0
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
        .filter(|w|
                w.chars().all(|ch| allowed.contains(ch))
        ).count() as i32
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
    let (mut arr1, arr2) = nums.iter().skip(2).fold(((vec![nums[0]], vec![nums[1]]), (0, 0)),
        |((mut arr1, mut arr2), (mut n1, mut n2)), &i|
        {
            if arr1[n1] > arr2[n2] {
                arr1.push(i); n1 += 1;
            } else {
                arr2.push(i); n2 += 1;
            } ((arr1, arr2), (n1, n2))
        }).0;
    arr1.extend(arr2);
    arr1
}

// <=======================================================================>

#[allow(unused)]
fn find_and_replace_pattern(words: Vec<String>, pattern: String) -> Vec<String> {
    fn is_match(word: &str, pattern: &str) -> bool {
        let mut w2pmap = HashMap::new();
        let mut p2wmap = HashMap::new();

        let ch_word = word.chars().collect::<Vec<char>>();
        let ch_patt = pattern.chars().collect::<Vec<char>>();

        for i in 0..ch_word.len() {
            if let Some(&prevw) = p2wmap.get(&ch_patt[i]) {
                if prevw != ch_word[i] { return false; }
            } else { p2wmap.insert(ch_patt[i], ch_word[i]); }

            if let Some(&prevp) = w2pmap.get(&ch_word[i]) {
                if prevp != ch_patt[i] { return false; }
            } else { w2pmap.insert(ch_word[i], ch_patt[i]); }
        } true
    }

    words.into_iter().filter(|word| is_match(word, &pattern)).collect()
}

// <=======================================================================>

pub fn can_be_typed_words(text: String, bale: String) -> i32 {
    text.split_whitespace()
        .filter(|w|
            bale.chars()
                .all(|c| !w.contains(c))
        ).count() as i32
}

// <=======================================================================>

pub fn count_prefixes(words: Vec<String>, s: String) -> i32 {
    words.iter()
        .filter(|&prefix| s.starts_with(prefix))
        .count() as i32
}

// <=======================================================================>

#[allow(unused)]
fn gcd1(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    } a
}

#[allow(unused)]
fn solution(arr: &[u64]) -> u128 {
    arr.iter().fold(arr[0], |acc, &x| gcd1(acc, x)) as u128 * arr.len() as u128
}

// <=======================================================================>

struct FT {
    tree:  Vec<usize>,
    treen: i32
}

impl FT {
    fn new(max: usize) -> FT {
        FT {
            tree:  vec![0; max + 2],
            treen: (max + 2) as i32
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
        } ans
    }
}

macro_rules! push {
    ($arr: ident <- $i: expr, $n: ident++, $ft: ident <- $id: ident) => {
        $arr.push($i); $n += 1;
        $ft.update($id, 1);
    };
}

pub fn result_array(nums: Vec<i32>) -> Vec<i32> {
    let n = nums.len();

    let (mut arr1, mut arr2) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut n1,   mut n2  ) = (1, 1);
    let (mut map,  mut mapn) = (HashMap::with_capacity(n), 0);

    let mut sorted = nums.clone(); sorted.sort_unstable();
    sorted.iter().for_each(|i| {
        map.entry(i).or_insert(mapn + 1); mapn += 1;
    });

    let (mut ft1, mut ft2) = (FT::new(mapn as usize), FT::new(mapn as usize));
    arr1.push(nums[0]);
    ft1.update(*map.get(&nums[0]).unwrap(), 1);

    arr2.push(nums[1]);
    ft2.update(*map.get(&nums[1]).unwrap(), 1);
    nums.iter().skip(2).for_each(|&i| {
        let id  = *map.get(&i).unwrap();
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
        } println!("c: {c}, i: {i}, dp: {dp:?}");
    } println!();

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
            } println!("i: {i}, j: {j}, v: {v}, c: {c}, t: {t}, dp[i] = {:?}, dp[j] = {:?}", dp[i], dp[j]);
        }
    } t
}

// <=======================================================================>

pub fn sum_of_encrypted_int(n: Vec<i32>) -> i32 {
    n
    .iter()
    .map(|&x| {
        let mut len = 0;
        let mut max = 0;
        let mut y   = x;
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

pub fn is_substring_present(s: String) -> bool {
    let x = s.chars().rev().collect::<String>();
    for i in 0..s.len() - 1 {
        if x.contains(&s[i..i + 2]) {
            return true;
        }
    } false
}

// <=======================================================================>

pub fn is_substring_present1(s: String) -> bool {
    (0..s.len() - 1).any(|i| s.chars().rev().collect::<String>().contains(&s[i..i + 2]))
}

// <=======================================================================>

pub fn find_minimum_operations(s1: String, s2: String, s3: String) -> i32 {
    let (x, y, z) = (s1.len(), s2.len(), s3.len());
    let (mut i, m) = (0, x.min(y).min(z));

    while i < m
    && &s1[i..=i] == &s2[i..=i]
    && &s2[i..=i] == &s3[i..=i] {
        i += 1
    }

    match i {
        0 => -1,
        _ => (x + y + z - 3 * i) as i32
    }
}

// <=======================================================================>

pub fn count_substrings(s: String, c: char) -> i64 {
    let m = s.chars().filter(|&ch| ch == c).count() as i64;
    m * (m + 1) / 2
}

// <=======================================================================>

pub fn minimum_abs_difference(mut arra: Vec<i32>) -> Vec<Vec<i32>> {
    arra.sort_unstable();
    let m = arra.windows(2).map(|p| p[1] - p[0]).min().unwrap();
    arra.windows(2)
        .filter_map(|p| if p[1] - p[0] == m {
            Some(vec![p[0], p[1]])
        } else {
            None
        }).collect::<Vec<_>>()
}

// <=======================================================================>

pub fn find_max_k(mut nums: Vec<i32>) -> i32 {
    nums.sort_unstable();
    nums.iter().rev().filter_map(|i| {
         if nums.binary_search(&-i).is_ok() {
            Some(*i)
        } else { 
            None 
        }
    }).max().unwrap_or(-1)
}

// <=======================================================================>

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
    } -1
}

// <=======================================================================>

pub fn minimum_deletions(word: String, k: i32) -> i32 {
    let mut map = [0; 26];
    let mut ans = i32::MAX;

    for c in word.chars() {
        map[c as usize - 97] += 1;
    }
    
    for &x in map.iter() {
        let high = x + k;
        ans = ans.min
        (
            map.iter().map(|&y| {
                if y > high   { y - high }
                else if y < x { y }
                else          { 0 }
            }).sum()
        );
    } ans
}

// <=======================================================================>

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

fn maximum_length_substring(s: String) -> i32 {
    s
    .chars()
    .enumerate()
    .fold((std::collections::HashMap::<u8, usize>::new(),
           0, 0), |(mut f, a, mut j), (i, c)|
    {
        *f.entry(c as u8).or_insert(0) += 1;
        while j < i && f[&(c as u8)] > 2 {
            if let Some(cnt) = f.get_mut(&s.as_bytes()[j]) {
                *cnt -= 1;
            } j += 1;
        }
        (f, a.max(i - j + 1), j)
    }).1 as i32
}

// <=======================================================================>

#[allow(unused)]
macro_rules! tovstr {
    ($($str: expr), *) => {
        vec![$($str.to_owned()), *]
    };
}

fn main() {
    dbg!(maximum_length_substring("aaaa".to_owned()));
    dbg!(minimum_deletions("aabcaba".to_owned(), 0));
    dbg!(minimum_abs_difference(vec![4, 2, 1, 3]));
    dbg!(count_substrings("abada".to_owned(), 'a'));
    dbg!(find_minimum_operations("abc".to_owned(), "abb".to_owned(), "ab".to_owned()));
    dbg!(is_substring_present1("abcd".to_owned()));
    dbg!(sum_of_encrypted_int(vec![10, 21, 31]));
}
