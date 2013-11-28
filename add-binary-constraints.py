import csv

def loadConstraints(filename):
    with open(filename, 'r') as f:
        data = [row for row in csv.reader(f.read().splitlines())]
    return data
binaryConstraints = loadConstraints('binary-constraints.csv')

print 'Our binary potentials indicate that a tweet cannot have the following labels together:'
for constraint in binaryConstraints:
	print constraint


# Note: enable the following once we have our binary constraints settled:
# print 'Adding binary constraints'
# for constraint in binaryConstraints:
# 	label1 = constraint[0]
# 	label2 = constraint[1]
# 	# We add a constraint for each pair of labels in our constraint specifying that we cannot have both of them co-occurring (i.e. turned to 1)
# 	csp.add_binary_potential(label1, label2, lambda l1, l2 : score_vector[label1] + score_vector[label2] <= 1