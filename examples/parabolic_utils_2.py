def other_dims_as_x0_2(q_1_1,q_2_2,q_2_0,q_2_1,x_0): return [-q_2_1*x_0/(q_1_1*x_0 - q_2_0)]
def other_dims_as_x1_2(q_1_1,q_2_2,q_2_0,q_2_1,x_1): return [q_2_0*x_1/(q_1_1*x_1 + q_2_1)]
def k_mat0_2(q_1_1,q_2_2,q_2_0,q_2_1): return [2*q_1_1**2*q_2_0, q_1_1*(q_1_1*q_2_2 - 4*q_2_0**2 - q_2_1**2), -2*q_2_0*(q_1_1*q_2_2 - q_2_0**2 - q_2_1**2), q_2_0**2*q_2_2]
def k_mat1_2(q_1_1,q_2_2,q_2_0,q_2_1): return [q_1_1**3, 4*q_1_1**2*q_2_1, q_1_1*(q_1_1*q_2_2 + 2*q_2_0**2 + 5*q_2_1**2), 2*q_2_1*(q_1_1*q_2_2 + q_2_0**2 + q_2_1**2), q_2_1**2*q_2_2]
