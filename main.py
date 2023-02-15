## SET UP VARIABLES
import numpy as np
import matplotlib.pyplot as plt

E = 2e11 # 2.9e7      # Young's Modulus
nu = .27    #.3         # Poisson's Ratio
rho = 7850 #.283       # density

Lx = 1          # Length of plate in X direction
Ly = 1          # Length of plate in y direction
Tk = 0.01       # Plate thickness
Nx = 41         # Number of nodes in x direction
Ny = 21         # Number of nodes in y direction

## CALCULATE THE STRESS-STRAIN MATRIX BASED ON THE PLANE STRESS/STRAIN MODELS
# Stress-Strain matrix for plane stress model
C = (E/(1-nu**2))*np.array([[1, nu, 0],
                            [nu, 1, 0],
                            [0, 0, (1-nu)/2]])

## CREATE NODES AND ELEMENTS
ElemNum = int((Nx-1)*(Ny-1))    # Number of elements
Np = int(Nx*Ny)                 # Number of nodes
x = np.linspace(0,Lx, Nx)       # Create nodes in x direction, equally spaced
y = np.linspace(0,Ly,Ny)        # Create nodes in y direction, equally spaced
[X,Y] = np.meshgrid(x,y)        # Create meshgrid
# Tge X and Y coordinates of each node, the nodes are numbered in columns, like this:
# Ny   2*Ny  .      .      .       Nx*Ny
# .    .     .      .      .         .
# .    .     .      .      .         .
# .    .     .      .      .         .
# 2    .     .      .      .         .
# 1  Ny+1    .      .      .    (Nx-1)*Ny+1
NodesCoordinates = np.hstack((np.reshape(np.transpose(X),(Np,1)),np.reshape(np.transpose(Y),(Np,1)))) # The X and Y coordinates of each node
# Create an array with node numbers
NodeList = np.zeros((Np))
for node in range(Np):
    NodeList[node] = node+1
NodeList = np.transpose(np.reshape(NodeList,(Nx,Ny)))

# Create an array in which each row represents the 4 nodes constructing an element
# The Matrix is created column by column
ElemList = np.zeros((ElemNum,4))
ElemList[:,0] = np.reshape(NodeList[:Ny-1,:Nx-1],(1,ElemNum))
ElemList[:,1] = np.reshape(NodeList[:Ny-1,1:Nx],(1,ElemNum))
ElemList[:,3] = np.reshape(NodeList[1:Ny,:Nx-1],(1,ElemNum))
ElemList[:,2] = np.reshape(NodeList[1:Ny,1:Nx],(1,ElemNum))
ElemList = ElemList.astype('int')


## CREATE THE GLOBAL STIFFNESS MATRIX
# CREATE THE SHAPE FUNCTION, NATURAL DERIVATIVES, JACOBIAN MATRIX, GLOBAL DERIVATIVES
# Complete Gauss Quadrature method is used so we need four gauss points
gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
# Global stiffness matrix
# The element stiffness matrix has the dimension of (2*4,2*4) in which 4 is the number of ndoes per element.
# The factor of 2 corresponds to the number of DOF of each node.
# The global stiffness matrix should then have the dimension of (2*Np,2*Np), in which Np is the total of nodes.
Kg = np.zeros((2*Np,2*Np))

# Calculate the stiffness integral for each element
for num in range(ElemNum):
    # Ke is the stiffness matrix associated with the element
    Ke = np.zeros((8,8))
    # B is strain-displacement matrix
    B = np.zeros((3,8))
    # Read the nodes associated with the element
    NodesInElement = ElemList[num,:]
    GlobalStiffnessMatrixIndices = np.hstack((NodesInElement-1,NodesInElement+Np-1)).astype(int)

    # Find the coordinates of each node inside the element of interest.
    # Note that the following nodes should be connected.
    ElemNodes = np.array([X[np.where(NodeList == NodesInElement[0])], Y[np.where(NodeList == NodesInElement[0])],
                          X[np.where(NodeList == NodesInElement[1])], Y[np.where(NodeList == NodesInElement[1])],
                          X[np.where(NodeList == NodesInElement[2])], Y[np.where(NodeList == NodesInElement[2])],
                          X[np.where(NodeList == NodesInElement[3])], Y[np.where(NodeList == NodesInElement[3])]])
    ElemNodes = np.reshape(ElemNodes, (4, 2))

    # Calculate the required parameters per each gauss point
    for zeta in gauss_points:
        for eta in gauss_points:
            # Shape function
            ShapeFunction = 1/4*np.transpose(np.array([(1-zeta)*(1-eta), (1+zeta)*(1-eta), (1+zeta)*(1+eta), (1-zeta)*(1+eta)]))
            # Derivatives of the shape function wrt natural coordinates
            ShapeFunctionNaturalDerivative = 1/4*np.array([[-(1-eta), -(1-zeta)],
                                                           [(1 - eta), -(1 + zeta)],
                                                           [(1 + eta), (1 + zeta)],
                                                           [-(1+eta),(1-zeta)]])
            # Approximate ioparametric Jacobian matrix, d(global)/d(natural)
            J = np.matmul(np.transpose(ElemNodes),ShapeFunctionNaturalDerivative)
            Jinv=np.linalg.inv(J)
            # Derivatives of the shape function wrt global coordinates
            ShapeFunctionGlobalDerivatives = np.matmul(Jinv,np.transpose(ShapeFunctionNaturalDerivative))

            # Form the strain-displacement matrix
            # The displacement vector is saved as {u1,u2,u3,...,v1,v2,v3,...}
            B[0,:4] = ShapeFunctionGlobalDerivatives[0,:]
            B[1,4:] = ShapeFunctionGlobalDerivatives[1,:]
            B[2,:4] = ShapeFunctionGlobalDerivatives[1,:]
            B[2,4:] = ShapeFunctionGlobalDerivatives[0,:]

            Ke = Ke + Tk*np.matmul(np.matmul(np.transpose(B),C),B)*np.linalg.det(J)

    Kg[tuple(np.meshgrid(GlobalStiffnessMatrixIndices,GlobalStiffnessMatrixIndices))] += Ke


## SET UP BOUNDARY CONDITIONS AND FORCES
UV = np.zeros((2*Np,1))
# Two nodes at the bottom are anchored. So need to find the indices associated with those nodes
# Use np.intersect1d command if there are multiple constraints on the geometry

# BC1 = np.intersect1d(np.where(NodesCoordinates[:,0]==0),np.where(NodesCoordinates[:,1]==0))
# BC2 = np.intersect1d(np.where(NodesCoordinates[:,0]==Lx),np.where(NodesCoordinates[:,1]==0))
BC1 = np.array(np.where(NodesCoordinates[:,0]==0))

# R is the vector of forces. Each node can undergo force in two directions: Rx and Ry
# Fx is a distributed force (psi/in) to avoid any singularity in the model
Fx = 5e7
R = np.zeros((2*Np,1))
# ForceNodes = np.array(np.where(NodesCoordinates[:,0]==0))
ForceNodes = np.array(np.where(NodesCoordinates[:,0]==Lx))

R[ForceNodes]=Fx*(Ly/(Ny-1))
R[ForceNodes[:,0]]=R[ForceNodes[:,-1]]=Fx*(Ly/(Ny-1))/2


## Solve for displacement
# There is no need to solve the U = K**-1 * R equation for the boundary conditions.
# Because I already know the value of U at the boundary conditions.
# So I define the active nodes as the nodes that are not a boundary condition
# ActiveNodes = np.setdiff1d(range(2*Np),[BC1,BC1+Np,BC2, BC2+Np])
ActiveNodes = np.setdiff1d(range(2*Np),[BC1,BC1+Np])
UV[ActiveNodes]=np.matmul(np.linalg.inv(Kg[tuple(np.meshgrid(ActiveNodes,ActiveNodes))]),R[ActiveNodes])


## Post Processing for strain and stress
# There are several methods to calculate strain and stress. For now, I have decided to calculate these values at the
# gauss points. One of the alternatives is to calculate strain and stress at the nodes.
# This part of the code is a bit redundant as I am re-calculating B matrix for each element, similar to previous steps.
# But I leave it as is if someday I decide to use the nodes (instead of gauss points) to calculate stress and strains at

e_xx = np.zeros((ElemNum,4))
e_yy = np.zeros((ElemNum,4))
e_xy = np.zeros((ElemNum,4))

s_xx = np.zeros((ElemNum,4))
s_yy = np.zeros((ElemNum,4))
s_xy = np.zeros((ElemNum,4))

NaturalNodes = [-1, 1]
for num in range(ElemNum):
    # Read the nodes associated with the element
    NodesInElement = ElemList[num,:]

    ElemNodes = np.array([X[np.where(NodeList == NodesInElement[0])], Y[np.where(NodeList == NodesInElement[0])],
                          X[np.where(NodeList == NodesInElement[1])], Y[np.where(NodeList == NodesInElement[1])],
                          X[np.where(NodeList == NodesInElement[2])], Y[np.where(NodeList == NodesInElement[2])],
                          X[np.where(NodeList == NodesInElement[3])], Y[np.where(NodeList == NodesInElement[3])]])
    ElemNodes = np.reshape(ElemNodes, (4, 2))

    ElemDOFs = np.reshape(np.array([NodesInElement-1, NodesInElement-1+Np]),(1,8)).astype('int')
    ElemDOFs = np.reshape([NodesInElement-1, NodesInElement-1+Np],(1,8)).astype('int')

    B = np.zeros((3,8))
    # To assign the values of strain and stress to the correct node within each element
    Nodecounter = 0
    # The loop reads the Gauss points in the following order: [-1,-1], [1,-1], [1,1], [-1,1], which matches the order in
    # which the nodes of element is read. This is to ensure that the strain/stress values correspond to the correct node
    NaturalNodes =[[-1,-1],[1,-1],[1,1],[-1,1]]
    for node in NaturalNodes:
        zeta = node[0]
        eta = node[1]
        # There is no need to define shape function and shape function derivatives again. I keep these definitions
        # in case, a different set of points (e.g. nodes instead of gauss points) are used for processing.
        ShapeFunction = 1/4*np.transpose(np.array([(1-zeta)*(1-eta), (1+zeta)*(1-eta),
                                                   (1+zeta)*(1+eta), (1-zeta)*(1+eta)]))
        # Derivatives of the shape function wrt natural coordinates
        ShapeFunctionNaturalDerivative = 1/4*np.array([[-(1-eta), -(1-zeta)],
                                                       [(1 - eta), -(1 + zeta)],
                                                       [(1 + eta), (1 + zeta)],
                                                       [-(1+eta),(1-zeta)]])
        # Approximate ioparametric Jacobian matrix, d(global)/d(natural)
        J = np.matmul(np.transpose(ElemNodes),ShapeFunctionNaturalDerivative)
        Jinv=np.linalg.inv(J)
        # Derivatives of the shape function wrt global coordinates
        ShapeFunctionGlobalDerivatives = np.matmul(Jinv,np.transpose(ShapeFunctionNaturalDerivative))

        # Form the strain-displacement matrix
        # The displacement vector is saved as {u1,u2,u3,...,v1,v2,v3,...}
        B[0,:4] = ShapeFunctionGlobalDerivatives[0,:]
        B[1,4:] = ShapeFunctionGlobalDerivatives[1,:]
        B[2,:4] = ShapeFunctionGlobalDerivatives[1,:]
        B[2,4:] = ShapeFunctionGlobalDerivatives[0,:]

        # Reading the displacement associated with the element from matrix UV which is already solved for.
        # For some reason, I end up with 3D matrix. So I just grab the first two dimensions.
        # The components are of the strain matrix of the element is [e_xx, e_yy, 2*e_xy]
        # The components are of the stress matrix of the element is [sigma_xx, sigma_yy, tau_xy]

        SElem = np.matmul(B,UV[ElemDOFs])
        StrainElem = SElem[:,:,0]
        StressElem = np.matmul(C,np.transpose(StrainElem))

        # Assign the values
        e_xx[num,Nodecounter] = StrainElem[0,0]
        e_yy[num,Nodecounter] = StrainElem[0,1]
        e_xy[num,Nodecounter] = StrainElem[0,2]

        s_xx[num,Nodecounter] = StressElem[0,0]
        s_yy[num,Nodecounter] = StressElem[1,0]
        s_xy[num,Nodecounter] = StressElem[2,0]
        Nodecounter = Nodecounter + 1

# Calculate the average stress/strain of each node
NodeStrain_xx = np.zeros((Np,1))
NodeStrain_yy = np.zeros((Np,1))
NodeStrain_xy = np.zeros((Np,1))
NodeStress_xx = np.zeros((Np,1))
NodeStress_yy = np.zeros((Np,1))
NodeStress_xy = np.zeros((Np,1))

# Takes the average of stress/strain values associated to the node of interest
for node in range(Np):
    NodeStrain_xx[node] = np.mean(e_xx[np.where(ElemList==node+1)])
    NodeStrain_yy[node] = np.mean(e_yy[np.where(ElemList==node+1)])
    NodeStrain_xy[node] = np.mean(e_xy[np.where(ElemList==node+1)])
    NodeStress_xx[node] = np.mean(s_xx[np.where(ElemList==node+1)])
    NodeStress_yy[node] = np.mean(s_yy[np.where(ElemList==node+1)])
    NodeStress_xy[node] = np.mean(s_xy[np.where(ElemList==node+1)])

# Calculate the Von-Mises Stress for plane Stress model
sigma_vm = np.sqrt((NodeStress_xx*NodeStress_xx)-(NodeStress_xx*NodeStress_yy)+(NodeStress_yy*NodeStress_yy) +
                   (3*NodeStress_xy*NodeStress_xy))


## Plot the results - Displacement
# The first half of UV vector is the u component of the displacement and the second half is the v component.
u = UV[:Np]
uMat = np.reshape(u,(Nx,Ny))
v = UV[Np:]
vMat = np.reshape(v,(Nx,Ny))

plotU = plt.figure(1)
plt.contourf(X,Y,np.transpose(uMat),15, cmap= 'rainbow')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$u_{x}$')
plt.colorbar(ticks=np.linspace(np.min(uMat), np.max(uMat), 15, endpoint=True))

plotV = plt.figure(2)
plt.contourf(X,Y,np.transpose(vMat), cmap= 'rainbow')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$u_{y}$')
plt.colorbar(ticks=np.linspace(np.min(vMat), np.max(vMat), Ny//2, endpoint=True))

## Plot the results - Von-Mises Stress
# You can also plot normal strains (NodeStrain_xx, _yy, _xy) and stresses, (NodeStress_xx, _yy, _xy).
# You need to reshape them to Nx*Ny dimension.

sigma_vm_mat = np.reshape(sigma_vm,(Nx,Ny))
plotStress = plt.figure(3)
plt.contourf(X,Y,np.transpose(sigma_vm_mat),15, cmap= 'rainbow')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title(r'$\sigma_{v}$')
plt.colorbar(ticks=np.linspace(np.min(sigma_vm_mat), np.max(sigma_vm_mat), 15, endpoint=True))
plt.show()


