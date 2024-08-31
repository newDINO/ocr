fn main() {

}

enum MathNode {
    Add(AddNode),
    Sub(SubNode),
    Uint(usize),
    Neg(NegNode),
}
impl MathNode {
    fn to_latex(&self) -> String {
        match self {
            MathNode::Add(node) => node.to_latex(),
            MathNode::Sub(node) => node.to_latex(),
            MathNode::Uint(num) => num.to_string(),
            MathNode::Neg(node) => node.to_latex(),
        }
    }
}

struct ValNode {
    
}

struct NegNode {
    inner: Box<MathNode>
}
impl NegNode {
    fn to_latex(&self) -> String {
        match &*self.inner {
            MathNode::Sub(_) | MathNode::Add(_) => "-(".to_owned() + &self.inner.to_latex() + ")",
            _ => "-".to_owned() + &self.inner.to_latex()
        }
    }
}


struct AddNode {
    left: Box<MathNode>,
    right: Box<MathNode>,
}
impl AddNode {
    fn to_latex(&self) -> String {
        let right_latex = match &*self.right {
            MathNode::Neg(_) => "(".to_owned() + &self.right.to_latex() + ")",
            _ => self.right.to_latex()
        };
        self.left.to_latex() + "+" + &right_latex
    }
}

struct SubNode {
    left: Box<MathNode>,
    right: Box<MathNode>,   
}
impl SubNode {
    fn to_latex(&self) -> String {
        let right_latex = match &*self.right {
            MathNode::Sub(_) | MathNode::Add(_) | MathNode::Neg(_) => "(".to_owned() + &self.right.to_latex() + ")",
            _ => self.right.to_latex()
        };
        self.left.to_latex() + "-" + &right_latex
    }
}