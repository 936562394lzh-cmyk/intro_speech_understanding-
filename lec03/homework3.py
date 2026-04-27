
def words2characters(words):
    """
    This function converts a list of words into a list of characters.

    @param:
    words - a list of words

    @return:
    characters - a list of characters

    Every element of "words" should be converted to a str, then split into
    characters, each of which is separately appended to "characters." For
    example, if words==['hello', 1.234, True], then characters should be
    ['h', 'e', 'l', '1', 'o', '1', '.', '2', '3', '4', 'T', 'r', 'u', 'e']
    """

    characters = []
    for item in words:
        # Convert the item to a string
        s = str(item)
        # Append each character individually
        for ch in s:
            characters.append(ch)
    return characters
