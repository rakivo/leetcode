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

fn main() {}
