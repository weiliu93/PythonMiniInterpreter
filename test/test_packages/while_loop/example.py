value = 0
step = 1
while value < 100:
    print(value)
    value += step
    if value % 2 == 0:
        continue
    step *= 2
    if value > 90:
        break
