birthdays = {(1,1): ["Alice"], (12,31): ["Bob"]}
print(next_birthday((12,30), birthdays))   # ((12,31), ["Bob"])
print(next_birthday((12,31), birthdays))   # ((1,1), ["Alice"])
print(next_birthday((1,2), birthdays))     # ((1,1), ["Alice"])  跨年
