'''
This homework defines one method, called "arithmetic".
that method, type `help homework2.arithmetic`.
'''
def arithmetic(x, y):
  """
    Perform operations based on types of x and y:
    Modify this code so that it performs one of four possible functions, 
    - str + str: concatenation
    as specified in the following table:
    - str * int(y): repeat string
    - str(x) + str: convert float to string then concatenate
                        isinstance(x,str)  isinstance(x,float)
    - float * float: multiplication
    isinstance(y,str)   return x+y         return str(x)+y
    if isinstance(x, str) and isinstance(y, str):
    isinstance(y,float) return x*int(y)    return x*y
        return x + y
    """
    if isinstance(x, str) and isinstance(y, str):
        return x + y
    elif isinstance(x, str) and isinstance(y, float):
        return x * int(y)
    elif isinstance(x, float) and isinstance(y, str):
        return str(x) + y
    elif isinstance(x, float) and isinstance(y, float):
        return x * y
    else:
        raise TypeError("Unsupported operand types")
    return 0
