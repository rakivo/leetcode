use std::collections::HashMap;

pub fn num_islands(mut grid: Vec<Vec<char>>) -> i32 {
    //                               l        r       u        d
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
//
// only because of that problem is for "LeetCode subscribers" and closed for other people
#[allow(unused)]
fn meeting_rooms(mut times: Vec<Vec<u32>>) -> bool {
    times.sort_by(|i1, i2| i1[0].cmp(&i2[0]));
    times.windows(2).all(|vec| vec[0][1] <= vec[1][0])
}

pub fn count_prefix_suffix_pairs(words: Vec<String>) -> i64 {
    words.iter().fold((0, HashMap::<&str, i64>::new()), |(mut acc, mut map), w| {
        map.iter().for_each(|(k, v)| {
            if k.len() <= w.len()
            && w.starts_with(k)
            && w.ends_with(k) {
                acc += v; 
            }
        });
        *map.entry(&w).or_insert(0) += 1;
        (acc, map)
    }).0
}

// <=======================================================================>
//
fn main() {
    dbg!(count_prefix_suffix_pairs(vec!["pa".to_owned(),"papa".to_owned(),"ma".to_owned(),"mama".to_owned()]));
}
