int maximum_value(const int *items, int count) {
    int best = items[0];
    for (int index = 1; index < count; ++index) {
        if (items[index] > best) best = items[index];
    }
    return best;
}
