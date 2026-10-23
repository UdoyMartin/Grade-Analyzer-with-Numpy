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