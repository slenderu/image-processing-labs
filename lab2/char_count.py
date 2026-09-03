import sys

filename = sys.argv[1]
with open(filename, "rb") as f:
    data = f.read()

print(filename, "contains", len(data), "bytes")
for b in sorted(set(data)):
    print("byte", b, "= hex", format(b, "02x"), "=", repr(chr(b)), "occurs", data.count(b), "times")