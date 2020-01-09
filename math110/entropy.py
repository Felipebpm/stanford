import math

sentence = "istheoutcomeofauniformrandomEnglishletterfromthissentence"
letters = "abcdefghijklmnopqrstuvwxyz"

occurrences = {}

for letter in letters:
    occurrences[letter] = 0


final = sentence.lower()
print(final)
for letter in final:
    if letter != ' ':
        occurrences[letter] += 1
print (occurrences)

entropy = 0
sample = len(sentence)
print(sample)

# calculate entropy
for letter in letters:
    if occurrences[letter] != 0:
        entropy += ((occurrences[letter] / sample) * math.log(sample / occurrences[letter], 2))

print("entropy: " + str(entropy))
print("bound: " + str(math.log(sample, 2)))
