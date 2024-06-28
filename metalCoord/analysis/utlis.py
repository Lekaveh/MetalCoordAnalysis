
def elementCode(code):
    """
    Returns a string containing only the uppercase letters from the input code.

    Parameters:
    code (str): The input code.

    Returns:
    str: A string containing only the uppercase letters from the input code.
    """
    result = []
    for x in code:
        if x.isupper():
            result.append(x)
        else:
            result[-1] += x
    return "".join(dict.fromkeys(result))


def elements(code):
    """
    Extracts individual elements from a given code.

    Args:
        code (str): The code to extract elements from.

    Returns:
        list: A list of individual elements extracted from the code.
    """
    result = []
    for x in code:
        if x.isupper():
            result.append(x)
        else:
            result[-1] += x
    return result