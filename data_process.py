with open("output.csv", "w") as output:

    for G in range(1,4):
        for L in range(1,4):
            exp = [f"{L}x{G}"]
            for R in range(3):
                with open(f"{L}x{G}_{R}.txt", "r") as f:
                    value = eval(f.read().strip())
                    if isinstance(value, complex):
                        value = value.real
                    exp.append(value)
            output.write(",".join(map(str, exp)) + "\n")