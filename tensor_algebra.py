from sympy import simplify, symbols
from functools import reduce

class TensorProduct:
    '''
    Building tensor algebra. 
    '''

    def __new__(cls, left, right):
        '''
        We need to be able to have associative tensor algebra.
        To do this, we check 4 instances
        (tensor expression, tensor)
        (tensor, tensor expression)
        (tesnor expression, tensor expression)
        (tensor, tensor)
        Tensor expression is an assortment of sums of tensors (a + b)
        Tensor is just a single tensor (c)
        Hence; for example suppose we have (a + b) ⊗ (c)
        We want to return (a ⊗ c) + (b ⊗ c)
        '''
        if isinstance(left, TensorExpression) or isinstance(right, TensorExpression):
            left_expr = left if isinstance(left, TensorExpression) else TensorExpression([(1, left)])
            right_expr = right if isinstance(right, TensorExpression) else TensorExpression([(1, right)])
            terms = []
            for c1, tp1 in left_expr.terms:
                for c2, tp2 in right_expr.terms:
                    new_coeff = simplify(c1 * c2)
                    new_tp = TensorProduct(tp1, tp2)
                    terms.append((new_coeff, new_tp))
             
            return TensorExpression(terms)
        

        self = super().__new__(cls)
        self.left = left
        self.right = right
        return self

    def __repr__(self): 
        return f"{self.left} ⊗ {self.right}"
    
    def __eq__(self, other):
        '''
        Equality.
        '''
        return (self.left, self.right) == (other.left, other.right)
    
    def __hash__(self):
        return hash((self.left, self.right))
    
    def __rmul__(self, scalar):
        '''
        Define scalar multiplication.
        '''
        return TensorExpression([(simplify(scalar), self)])
    
    def __mul__(self, other):
        '''
        Define tensor multiplication.
        '''
        return TensorProduct(self, other)

    def __add__(self, other):
        '''
        Define tensor addition.
        '''
        return TensorExpression([(1, self)]) + TensorExpression([(1, other)])
    
   
class TensorExpression:
    '''
    Tensor Expression.
    '''

    def __init__(self, terms = None):
        self.terms = terms if terms else []

    def __add__(self, other):
        if isinstance(other, TensorProduct):
            other = TensorExpression([(1, other)])

        return(TensorExpression(self.terms + other.terms))
    
    def __rmul__(self, scalar):
        new_terms = [(simplify(scalar * coeff), tp) for coeff, tp in self.terms]
        return TensorExpression(new_terms)
    
    def __mul__(self, other):

        if isinstance(other, TensorExpression):
            new_terms = []
            for c1, tp1 in self.terms:
                for c2, tp2 in other.terms:
                    new_coeff = simplify(c1 * c2)
                    new_tp = tp1 * tp2
                    new_terms.append((new_coeff, new_tp))
            return TensorExpression(new_terms)
        
        elif isinstance(other, TensorProduct):
            return self * TensorExpression([(1, other)])
        
        else:
            raise TypeError("Unsupported type for tensor product")

    def __repr__(self):
        return " + ".join(f"{coeff}·({tp})" for coeff, tp in self.terms)
    

def flatten_tensor(tp):

    if isinstance(tp, TensorProduct):
        return flatten_tensor(tp.left) + flatten_tensor(tp.right)
    else:
        return [tp]
    
def rebuild_tensor(elements):
    return reduce(lambda x, y: TensorProduct(x, y), elements)

def insert_tensor(tensor_chain, new_tensor, index):

    if isinstance(tensor_chain, TensorProduct):
        tensor_chain = TensorExpression([(1, tensor_chain)])

    new_terms = []
    for coeff, tp in tensor_chain.terms:
        flat = flatten_tensor(tp)
        flat.insert(index, new_tensor)
        new_tp = rebuild_tensor(flat)
        new_terms.append((coeff, new_tp))

    return TensorExpression(new_terms)

### tests

# (a ⊗ b) tensor
# (a + b) ⊗ (c + d) = (a ⊗ c) + (a ⊗ d) + (b ⊗ c) + (b ⊗ d) associativity

q = symbols('q')
e_1 = symbols('e_1') ## e1: (e_{1})
e_2 = symbols('e_2')
de_1 = symbols('de_1') ## dual e1: (e^{1})
de_2 = symbols('de_2')
V = symbols('V')
dV = symbols('dV')
 
tester = ['(e_1) ⊗ (de_1) + (e_2) ⊗ (de_2)', '(e_1) ⊗ (de_1) + (e_2) ⊗ (de_2)']
element_1 = TensorProduct(e_1, de_1) + TensorProduct(e_2, de_2)
element_2 = TensorProduct(e_1, de_1) + TensorProduct(e_2, de_2)

print(isinstance(element_1, TensorExpression))
print(isinstance(element_2, TensorExpression))

element_3 = TensorProduct(element_1, element_2)

print(element_3)
print(isinstance(element_3, TensorExpression))

print(insert_tensor(element_3, element_2, index=1))