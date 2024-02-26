from functools import reduce

dial_kbs = [
    {
        'name': 'xiaoming',
        'age': '25'
    },
    {
        "name": 'zhangsan',
        "age": '25'
    }
]

dial_kb_set = set(list(reduce(lambda a, b: a + b, [list(kb_dict.values()) for kb_dict in dial_kbs])))
print(dial_kb_set)