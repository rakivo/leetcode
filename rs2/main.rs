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

fn main() {}
