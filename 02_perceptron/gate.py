def AND(x1, x2):
    r = x1 * 0.5 + x2 * 0.5 + (-0.7)
    if r > 0:
        return 1
    else:
        return 0


def OR(x1, x2):
    r = x1 * 0.5 + x2 * 0.5 + (-0.3)
    if r > 0:
        return 1
    else:
        return 0


def NAND(x1, x2):
    r = x1 * -0.5 + x2 * -0.5 + 0.7
    if r > 0:
        return 1
    else:
        return 0


def XOR(x1, x2):
    r1 = NAND(x1, x2)
    r2 = OR(x1, x2)
    return AND(r1, r2)


def test(func):
    print("------------")
    print(func(0, 0))
    print(func(0, 1))
    print(func(1, 0))
    print(func(1, 1))
    print("------------")


test(AND)
test(OR)
test(NAND)
test(XOR)
