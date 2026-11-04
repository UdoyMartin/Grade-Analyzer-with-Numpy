import numpy as np

scores=np.array([
    [85, 90, 78, 88],
    [92, 88, 95, 91],
    [70, 65, 80, 72],
    [60, 55, 68, 58],
    [95, 97, 92, 98],
])
subjects=['Math', 'Science', 'English', 'History']
students=['Jadu','Elon','Steve','Tony','Peter']

#np.array, np.zeros, np.ones, np,arange, np.linspace
print("\n scores(np.array): \n",scores)
print("\n np.zeros((5, 4)): \n",np.zeros((5, 4)))
print("\n np.ones((5, 4)): \n",np.ones((5, 4)))
print("\n np.arange(0, 100, 10): \n",np.arange(0, 100, 10))
print("\n np.linspace(0, 100, 5): \n",np.linspace(0, 100, 5))

#array insecption ( check shape, size, ndim, dtype)
print(f"\n shape: {scores.shape}")
print(f" size: {scores.size}")
print(f" ndim: {scores.ndim}")
print(f" dtype: {scores.dtype}")
print(f" Itemsize: {scores.itemsize}")

#Reshaping & Flattening  
reshaped = scores.reshape(4, 5)
print("\n Reshaped: \n",reshaped)

flat_ravel= scores.ravel()
print("\n Ravel: \n",flat_ravel)

flat_flatten= scores.flatten()
print("\n Flatten: \n",flat_flatten)

#indexing & slicing
print("\n Steve's scores (from row 0):",scores[2,:])
print(" Science (from col 0):",scores[:, 1])
#print(" Top 2 students from 2 subjects:",scores[:, 0])
#fancy indexing- pick specific students
print(" Pick specific students(Jadu, Steve, Peter):", scores[[0,2,4]])
print(" Find failing scores:", scores[scores<70])
fail_row, fail_col= np.where(scores<70)
for r, c in zip(fail_row, fail_col):
    print(f" {students[r]} failed in {subjects[c]} with score {scores[r,c]}")

#Math & Statistics
total=np.sum(scores, axis=1)
avg_sum=np.mean(scores,axis=1)
avg_sub=np.mean(scores,axis=0)
std_stu=np.std(scores,axis=1)
median=np.median(scores,axis=1)


print("\n Total scores of each student:", total)
print(" Average scores of each student:", np.round(avg_sum, 2))
print(" Average scores of each subject:", np.round(avg_sub, 2))
print(" Standard deviation of each student:", np.round(std_stu, 2))
print(" Median scores of each student:", median)
print(" Highest score of a student:", np.max(scores))
print(" Lowest score of a student:", np.min(scores))

#Array operations
bonus= scores +5
print("\n After 5 bonus marks: \n", bonus)
#after clip
curved= np.clip(bonus, 0, 100)
print("\n After clipping to 100: \n", curved)

#stacking & splitting
print("\n Vertical stacking: \n", np.vstack((scores, curved)))
print("\n Horizontal stacking: \n", np.hstack((scores, curved)))
print("\n Concatenate along axis 0: \n", np.concatenate((scores, curved), axis=0))

#datasplit
# 1. np.hsplit(): splits 4 columns into 2 equal parts (shape: 5x2 each)
h1, h2 = np.hsplit(scores, 2)
print("\n Horizontal split (np.hsplit):\nPart 1:\n", h1, "\nPart 2:\n", h2)

# 2. np.vsplit(): requires an equal row split, so we use stacked scores (10 rows -> 2 parts of 5 rows each)
stacked = np.vstack((scores, curved))
v1, v2 = np.vsplit(stacked, 2)
print("\n Vertical split (np.vsplit on stacked):\nPart 1:\n", v1, "\nPart 2:\n", v2)

# 3. np.split(): general split along axis 1 (columns) into 2 equal parts
s1, s2 = np.split(scores, 2, axis=1)
print("\n Split along axis 1 (np.split):\nPart 1:\n", s1, "\nPart 2:\n", s2)

#sorting & searching
print("\n Sorted scores of each student: \n", np.sort(scores, axis=1))
print("\n Sorted scores of each student: \n", np.argsort(scores, axis=1)) 
print("\n Searching for score 95: \n", np.array(np.where(scores==95)))
print("\n searching best student in each subject: \n", np.argmax(scores, axis=0))
print("\n searching worst student in each subject: \n", np.argmin(scores, axis=0))

#Linear Algebra
# 1. np.dot() and np.matmul(): calculate weighted final scores for 4 subjects
weights = np.array([0.25, 0.35, 0.20, 0.20])

print("\n Weighted scores (np.dot):", np.round( np.dot(scores, weights), 2))
print(" Weighted scores (np.matmul):", np.round(np.matmul(scores, weights), 2))

# 2. np.linalg.det() and np.linalg.inv(): square matrix operations on first 4 students x 4 subjects
square_scores = scores[:4, :]
determinant = np.linalg.det(square_scores)
inverse_matrix = np.linalg.inv(square_scores)
print("\n Determinant (np.linalg.det):", np.round(determinant, 2))
print(" Inverse Matrix (np.linalg.inv):\n", np.round(inverse_matrix, 4))

#Save and Load
print("\n Saving scores to 'scores.npy' \n", np.save('scores.npy', scores))
print("\n Loading scores from 'scores.npy;\n", np.load('scores.npy'))
print("\n Saving scores to 'scores.txt'", np.savetxt('scores.txt', scores, fmt='%d'))
print("\n Loading scores from 'scores.txt': \n", np.loadtxt('scores.txt', dtype=int))
print("\n Saving scores to 'scores.csv'", np.savetxt('scores.csv', scores, delimiter=',', fmt='%d'))