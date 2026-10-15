def group_by_initial(words):
    groups = {}
    for word in words:
        if word:
            groups.setdefault(word[0].lower(), []).append(word)
    return groups

assert group_by_initial(["apple", "Ant", "boat"]) == {"a": ["apple", "Ant"], "b": ["boat"]}
