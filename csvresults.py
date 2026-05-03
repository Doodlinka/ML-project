with open("results.txt") as inf:
    with open("results.csv", "w") as outf:
        for line in inf:
            outf.write("\t".join(line.split()) + "\n")