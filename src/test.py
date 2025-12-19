d = {"a": 10, "b": 5}

match d:
    case {"b": 5, **rest}:
        print("b=5 in ", d, rest)
