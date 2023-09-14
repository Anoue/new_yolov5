# n, m = map(int, (input().split()))
# s = []
# a = []
# for i in range(n):
#     name, force_value = input().split()
#     s.append(name)
#     a.append(int(force_value))

# result = ['Y'] * n


# def fighting(idx1, idx2, c1, c2):
#     if c1 == 'Y':
#         if a[idx1] > a[idx2]:
#             result[idx2] = 'N'
#         elif a[idx1] < a[idx2]:
#             result[idx1] = 'N'
#         else:
#             result[idx1] = 'N'
#             result[idx2] = 'N'
#     else:
#         if c2 == 'N':
#             return
#         else:
#             if a[idx1] > a[idx2]:
#                 result[idx2] = 'N'
#             else:
#                 return


# for i in range(m):
#     idx1, idx2, c1, c2 = input().split()
#     idx1 = int(idx1) - 1
#     idx2 = int(idx2) - 1
#     # print(idx1, idx2, c1, c2)
#     if result[idx1] == 'N' or result[idx2] == 'N':
#         continue
#     if s[idx1] == s[idx2]:
#         continue

#     elif s[idx1] == "human":
#         fighting(idx1, idx2, c1, c2)
#     else:
#         fighting(idx2, idx1, c2, c1)

# answer="".join(result)
# print(answer)
m = dict()
m['a'] = 1
print(m)

