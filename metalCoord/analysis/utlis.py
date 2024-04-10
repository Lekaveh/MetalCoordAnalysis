
def elementCode(code):
    result = []
    for x in code:
        if x.isupper():
            result.append(x)
        else:
            result[-1] += x
    return "".join(dict.fromkeys(result))


def elements(code):
    result = []
    for x in code:
        if x.isupper():
            result.append(x)
        else:
            result[-1] += x
    return result