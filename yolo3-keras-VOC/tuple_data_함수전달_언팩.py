
def compute_loss(target1, target2):
    print(target1, target2)

target = (('1',23), ('2',55), ('3',77))
#target = ('1',23)
for i in range(3):
    compute_loss(*target[i])