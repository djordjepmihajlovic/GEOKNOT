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
        (tensor expression, tensor expression)
        (tensor, tensor)
        Tensor expression is an assortment of sums of tensors (a + b)
        Tensor is just a single tensor (c)
        Hence; for example suppose we have (a + b) ⊗ (c)
        We want to return (a ⊗ c) + (b ⊗ c)
        '''

        if isinstance(left, UnitTensor):
            if isinstance(right, TensorExpression):
                self.expanded = right
                return
            self.left = right
            self.right = None
            self.expanded = None
            return

        if isinstance(right, UnitTensor):
            if isinstance(left, TensorExpression):
                self.expanded = left
                return
            self.left = left
            self.right = None
            self.expanded = None
            return
        
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
        '''
        Hashing.
        '''
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
        def format(coeff, tp):
            if coeff == 1:
                return f"({tp})"
            
            elif coeff == 0:
                return None
            
            elif isinstance(tp, UnitTensor):
                return f"{coeff}"
            
            else:
                return f"({coeff})·({tp})"
            
        formatted = [format(c, t) for c, t in self.terms]

        return " + ".join(f for f in formatted if f is not None)
    
class UnitTensor:
    def __repr__(self):
        return "1"
    
    def __eq__(self, other):
        return isinstance(other, UnitTensor)
    

def flatten_tensor(tp):

    if isinstance(tp, TensorProduct):
        return flatten_tensor(tp.left) + flatten_tensor(tp.right)
    else:
        return [tp]
    
def rebuild_tensor(elements):
    if not elements:
        return UnitTensor()
    
    return reduce(lambda x, y: TensorProduct(x, y), elements)

def insert_tensor(tensor_chain, new_tensor, index):

    if isinstance(tensor_chain, TensorProduct):
        tensor_chain = TensorExpression([(1, tensor_chain)])

    new_terms = []
    for coeff, tp in tensor_chain.terms:
        flat = flatten_tensor(tp)
        flat.insert(index, new_tensor)
        new_tp = rebuild_tensor(flat)

        if isinstance(new_tp, TensorExpression): ### Avoiding nested TensorExpression terms!
            for c, inner_tp in new_tp.terms:
                new_terms.append((simplify(coeff * c), inner_tp))
        
        else:
            new_terms.append((simplify(coeff), new_tp))

    return TensorExpression(new_terms)

def apply_R_matrix(tensor_chain, R_matrix, index):

    if isinstance(tensor_chain, TensorProduct):
        tensor_chain = TensorExpression([(1, tensor_chain)])

    new_terms = []
    for coeff, tp in tensor_chain.terms:
        flat = flatten_tensor(tp)
        left = flat[:index-1]
        switch = [flat[index-1], flat[index]]
        right = flat[index+1:]

        inter_tp = rebuild_tensor(switch)

        R_image = R_matrix.get(inter_tp)

        for new_coeff, rep in R_image.terms:
            full = left + flatten_tensor(rep) + right
            new_tp = rebuild_tensor(full)
            new_terms.append((simplify(coeff * new_coeff), new_tp))

    return TensorExpression(new_terms)

def apply_cap(tensor_chain, coev, index):
    
    if isinstance(tensor_chain, TensorProduct):
        tensor_chain = TensorExpression([(1, tensor_chain)])

    new_terms = []
    for coeff, tp in tensor_chain.terms:
        flat = flatten_tensor(tp)
        left = flat[:index-1]
        switch = [flat[index-1], flat[index]]
        right = flat[index+1:]

        inter_tp = rebuild_tensor(switch)
        coev_terms = coev.get(inter_tp)

        full = left + right
        new_tp = rebuild_tensor(full)
        new_terms.append((simplify(coeff * coev_terms), new_tp))

    return TensorExpression(new_terms)

q = symbols('q')
e_1 = symbols('e_1') ## e1: (e_{1})
e_2 = symbols('e_2')
de_1 = symbols('de_1') ## dual e1: (e^{1})
de_2 = symbols('de_2')
V = symbols('V')
dV = symbols('dV')

R_table_VV = {
    TensorProduct(e_1, e_1): q**(1/4)*TensorProduct(e_1, e_1),
    TensorProduct(e_1, e_2): q**(-1/4)*TensorProduct(e_2, e_1),
    TensorProduct(e_2, e_1): q**(-1/4)*TensorProduct(e_1, e_2) + (q**(1/4) - q**(-3/4))*TensorProduct(e_2, e_1),
    TensorProduct(e_2, e_2): q**(1/4)*TensorProduct(e_2, e_2),
}

R_table_VdV = {
    TensorProduct(e_1, de_1): q**(-1/2)*TensorProduct(de_1, e_1),
    TensorProduct(e_1, de_2): TensorProduct(de_2, e_1),
    TensorProduct(e_2, de_1): TensorProduct(de_1, e_2) + (q - q**(-1))*TensorProduct(de_2, e_1),
    TensorProduct(e_2, de_2): q**(-1/2)*TensorProduct(de_2, e_2),
}

R_table_dVV = {
    TensorProduct(de_1, e_1): q**(-1/2)*TensorProduct(e_1, de_1),
    TensorProduct(de_1, e_2): TensorProduct(e_2, de_1) + (q - q**(-1))*TensorProduct(e_1, de_2),
    TensorProduct(de_2, e_1): TensorProduct(e_1, de_2),
    TensorProduct(de_2, e_2): q**(-1/2)*TensorProduct(e_2, de_2),
}

coev_dVV = {
    TensorProduct(de_1, e_1): 1,
    TensorProduct(de_1, e_2): 0,
    TensorProduct(de_2, e_1): 0,
    TensorProduct(de_2, e_2): 1,
}

coev_VdV = {
    TensorProduct(e_1, de_1): q**(1/2),
    TensorProduct(e_1, de_2): 0,
    TensorProduct(e_2, de_1): 0,
    TensorProduct(e_2, de_2): q**(-1/2),
}

### Check a couple example scenarios to ensure this is working correctly

##################################
## Example scenario: RMI unknot ##
##################################

def check_RMI():
    # ev: initial instance of tensor C -> [V, dV]
    tensor = TensorProduct(e_1, de_1) + TensorProduct(e_2, de_2)
    # ev: insert ev at specific position (lets say left of original) -> [dV, V, V, dV]
    new_ev = q**(-1/2)*TensorProduct(de_1, e_1) + q**(1/2)*TensorProduct(de_2, e_2)
    print(tensor)
    print(new_ev)
    tensor = insert_tensor(tensor, new_ev, 0)
    print(tensor)
    # crossing: crossing occuring at specific position (lets say between V, dV [index:2]) -> [dV, V, V, dV]
    tensor = apply_R_matrix(tensor, R_table_VV, 2)
    print(tensor)
    # cap: cap off an ev at specific position (lets say dV, V) -> [V, dV]
    tensor = apply_cap(tensor, coev_dVV, 1)
    print(tensor)
    # cap: cap off last ev [V, dV] -> C
    tensor = apply_cap(tensor, coev_VdV, 1)
    print(tensor)

###################################
## Example scenario: RMII unknot ##
###################################

def check_RMII():
    # ev: initial instance of tensor C -> [V, dV]
    tensor = TensorProduct(e_1, de_1) + TensorProduct(e_2, de_2)
    # ev: insert ev at specific position (lets say left of original) -> [dV, V, V, dV]
    new_ev = q**(-1/2)*TensorProduct(de_1, e_1) + q**(1/2)*TensorProduct(de_2, e_2)
    tensor = insert_tensor(tensor, new_ev, 0)
    print(tensor)
    # crossing: crossing occuring at specific position (lets say between V, dV [index:2]) -> [dV, V, V, dV]
    tensor = apply_R_matrix(tensor, R_table_VV, 2)
    print(tensor)
    # cap: cap off an ev at specific position (lets say dV, V) -> [V, dV]
    tensor = apply_cap(tensor, coev_dVV, 1)
    print(tensor)
    # cap: cap off last ev [V, dV] -> C
    tensor = apply_cap(tensor, coev_VdV, 1)
    print(tensor)

####################################
## Example scenario: RMIII unknot ##
####################################

def check_RMIII():
    # ev: initial instance of tensor C -> [V, dV]
    tensor = TensorProduct(e_1, de_1) + TensorProduct(e_2, de_2)
    # ev: insert ev at specific position (lets say left of original) -> [dV, V, V, dV]
    new_ev = q**(-1/2)*TensorProduct(de_1, e_1) + q**(1/2)*TensorProduct(de_2, e_2)
    tensor = insert_tensor(tensor, new_ev, 0)
    print(tensor)
    # crossing: crossing occuring at specific position (lets say between V, dV [index:2]) -> [dV, V, V, dV]
    tensor = apply_R_matrix(tensor, R_table_VV, 2)
    print(tensor)
    # cap: cap off an ev at specific position (lets say dV, V) -> [V, dV]
    tensor = apply_cap(tensor, coev_dVV, 1)
    print(tensor)
    # cap: cap off last ev [V, dV] -> C
    tensor = apply_cap(tensor, coev_VdV, 1)
    print(tensor)

###############################
## Example scenario: trefoil ##
###############################

check_RMI()