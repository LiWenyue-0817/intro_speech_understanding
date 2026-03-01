'''
This homework defines one method, called "arithmetic".
that method, type `help homework2.arithmetic`.
'''

def arithmetic(x, y):
    """
    Modify this code so that it performs one of four possible functions, 
    as specified in the following table:

                        isinstance(x,str)  isinstance(x,float)
    isinstance(y,str)   return x+y         return str(x)+y
    isinstance(y,float) return x*int(y)    return x*y
    """
    # 判断y是字符串的情况
    if isinstance(y, str):
        if isinstance(x, str):
            return x + y
        elif isinstance(x, float):
            return str(x) + y
    # 判断y是浮点数的情况
    elif isinstance(y, float):
        if isinstance(x, str):
            return x * int(y)
        elif isinstance(x, float):
            return x * y
    # 若类型不匹配，返回0（保留原默认逻辑）
    return 0

# 测试用例（验证不同类型组合的结果）
if __name__ == "__main__":
    # 测试1: x=str, y=str → 字符串拼接
    print(arithmetic("hello", "world"))  # 预期: helloworld
    
    # 测试2: x=float, y=str → 浮点数转字符串+拼接
    print(arithmetic(3.14, " is pi"))   # 预期: 3.14 is pi
    
    # 测试3: x=str, y=float → 字符串重复int(y)次
    print(arithmetic("a", 3.0))         # 预期: aaa
    
    # 测试4: x=float, y=float → 浮点数相乘
    print(arithmetic(2.5, 4.0))         # 预期: 10.0
