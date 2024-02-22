use std::collections::{HashMap, HashSet};

struct RandomizedSet {
    n: usize,
    state: usize,
    map: HashMap<i32, usize>,
    values: [Option<i32>; 5000]
}

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

struct RandomizedCollection {
    map: HashMap<i32, (HashSet<usize>, usize)>,
    values: [Option<i32>; 10000],
    state: usize,
    n: usize
}

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

fn main() {
    let mut _rs = RandomizedSet::new();
    _rs.insert(1);
    _rs.insert(2);
    _rs.remove(2);
    _rs.get_random();

    let mut _rs = RandomizedCollection::new();
    _rs.insert(1);
    _rs.insert(2);
    _rs.remove(2);
    _rs.get_random();
}
