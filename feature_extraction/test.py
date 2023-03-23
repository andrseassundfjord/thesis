x_names = ['/feature/20', "/feature/22", "/feature/30"]
n = 5  # number of times to multiply the list
new_list = []
for i in range(n):
    for x in x_names:
        number = int(x.split('/')[-1]) + (16 * i)
        new_list.append(f"/feature/{number}")

print(len(new_list))
print(new_list)