from skimage.segmentation import mark_boundaries, slic, quickshift, watershed, felzenszwalb    
from tensorflow.keras.applications import ResNet50, VGG16



import copy
# perturbation argument should be array
def perturb_image(img, perturbation, segments,iter):
    global NFE
    active_pixels = np.where(perturbation == 1)[0]
    mask=np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image*mask[:,:,np.newaxis]
    pic=np.expand_dims(perturbed_image, axis=0)
    pred = model.predict(pic)[0][class_to_explain]
    ### fidelity check
    ##for label 1
    # pred = model.predict(pic)[0][class_to_explain][0]
    ### for label 2
    # pred = model.predict(pic)[0][class_to_explain][1]


    max_value=max(model.predict(pic)[0])
    k=model.predict(pic)[0]
    u=k.tolist()
    uu=u.index(max_value)
    Super=perturbation.sum(0)        
    ### fidelity check

    # for label 1 and non-fidelity

    if uu == class_to_explain[0][0]:
        ac=1
        
    else:
        ac=0
    

    
    
    if Super!=0:
         fit= [(nVar-Super+1)/nVar,pred]
    else:
         fit= [0,0]
    
    

    NFE=NFE+1  
    return pred[0],perturbed_image,ac,fit 

    

def SinglePointCrossover(x1,x2):
    import random
    import numpy as np
    nVar=len(x1)
    C=random.randint(1,nVar-1)
    y1=(x1[0:C]).tolist() + (x2[C:]).tolist()
    y2=(x2[0:C]).tolist() + (x1[C:]).tolist()
    return y1,y2

def Mutate(x):
    import random
    import numpy as np
    nVar=len(x)
    J=random.randint(0,nVar-1)
    y=copy.deepcopy(x)
    y[J]=1-x[J]
    return y,sum(y)



def Mutate2(x):
    import random
    import numpy as np
    nVar=len(x)
    J=[]
    s=np.trunc(nVar*0.1)
    for i  in range (int(s)):
       J.append(random.randint(0,nVar-1)) 
    y=copy.deepcopy(x)

    for i  in range (int(s)):
        y[J[i]]=1-x[J[i]]
       
    
    return y,sum(y)



def RouletteWheelSelection(P):
    r=random.uniform(0,1)
    c=np.cumsum(P)
    i=np.where(r<np.array(c))[0][0]
    return i
    

def Dominates(x,y):
    
    f1=False
    f2=False
    # flag=all(x >= y for i,j in (x,y))  and  any(x > y for i,j in (x,y))
    c1 = 0 
    c2 = 0
    for i in range(len(x)):
       if(x[i] >= y[i]):
           c1 += 1
       if(x[i] > y[i]):
           c2 += 1

    if(c1 == len(x)):
       f1=True
   
    if(c2 > 0):
       f2=True
   
    flag=f1 and f2   
    return flag

############# Non-dominated sorting function##################
def NonDominatedSorting(initial_pop):
    
    dFrame = pd.DataFrame(index=range(50),columns=range(1))
    nPop=len(initial_pop)
    for i in range(nPop):
        initial_pop[i].DominationSet=[]
        initial_pop[i].DominatedCount=0
        
    F=[]    
    for i in range (nPop):
        for j in range (i+1,nPop):
            p=initial_pop[i]
            q=initial_pop[j]
            if Dominates(p.Fit,q.Fit):
                # p.DominationSet=p.DominationSet.append(j)
                p.DominationSet.append(j)
                q.DominatedCount=q.DominatedCount+1
            if Dominates(q.Fit,p.Fit):
                # q.DominationSet=q.DominationSet.append(i)
                q.DominationSet.append(i)
                p.DominatedCount=p.DominatedCount+1                
            
            initial_pop[i]=p
            initial_pop[j]=q
        
        if initial_pop[i].DominatedCount==0:
          F.append(i)
          initial_pop[i].rank=0
    
    k=0
    dFrame.iloc[k,0]=F

    while(True):
        Q=[]
        for i in dFrame.iloc[k,0]:
            p=initial_pop[i]
            for j in p.DominationSet:
                q=initial_pop[j]
                q.DominatedCount=q.DominatedCount-1
                if q.DominatedCount==0:
                    Q.append(j)
                    q.rank=k+1
                initial_pop[j]=q
        if not Q:
            break;
            
        k=k+1
        dFrame.iloc[k,0]=Q
                    
    return initial_pop,F,dFrame


################  Calculate Distance######################

def CalCrowdingDistance(initial_pop, FrontsList):
    
 for s in range(int(FrontsList.shape[0])):  
   FitVector=[]
   for dd in range (int(len(FrontsList[0][s]))):
      FitVector.append(initial_pop[FrontsList[0][s][dd]].Fit) ###0=i
   nObj=len(FitVector[0])
   n=len(FitVector)
   d=np.zeros((n,nObj))
   for b in range (nObj):
      Fj=[]
      for l in range (n):
          Fj.append(FitVector[l][b])
      yy=np.sort(Fj)
      indexyy=(yy).argsort()[::-1]
      d[indexyy[0],b]=float('inf')
      for o in range (1,n-1):
         d[indexyy[o],b]=abs(yy[o+1]-yy[o-1])/abs(yy[0]-yy[-1])
      d[indexyy[-1],b]=float('inf')
   for i in range (n):
      initial_pop[FrontsList[0][s][i]].CrowdingDistance=np.sum(d[i,:])
         
         
 return initial_pop



################  Sort Population######################

def SortPopulation(initial_pop):
    
    # Sort based on crowding distance
    CD=[]
    RSO=[]
    for i in range (len(initial_pop)):
        CD.append(initial_pop[i].CrowdingDistance)
    # sorted(CD, reverse=True)
    CDSO=np.array(CD).argsort()[::-1]
    import operator
    initial_pop.sort(key=operator.attrgetter('CrowdingDistance'), reverse=True)
    # Sort based on ranks
    for i in range (len(initial_pop)):
        RSO.append(initial_pop[i].rank)
    RSO = np.array(RSO).argsort()
    initial_pop.sort(key=operator.attrgetter('rank'))
      
       
    FitVector = [obj.Fit for obj in initial_pop if obj.rank == 0 ]
    MaxAcc = [obj.pred for obj in initial_pop if obj.rank == 0 ]
    MaximumAcc=max(MaxAcc)
       
    return initial_pop, FitVector,MaximumAcc


##############BinaryTournamentSelection
def   BinaryTournamentSelection(pop):
    I = random.sample(list(range(len(pop))), 2)
    i1=I[0]
    i2=I[1]
    if pop[i1].rank < pop[i2].rank: 
        out=i1
    elif pop[i1].rank > pop[i2].rank:
        out=i2
    elif  pop[i1].CrowdingDistance > pop[i2].CrowdingDistance:
        out=i1
    else:
        out=i2
        
    return out       
    
##############PlotFit Function
def PlotFit(popu):
  fi=popu

  x = [item[0] for item in fi]
  y = [item[1][0] for item in fi]  # Access the first element of the NumPy array

  plt.scatter(x, y, marker='o', color='blue')  # Use scatter plot with circle markers  plt.xlabel('X-Axis Label')  # Replace with your desired label
  plt.ylabel('accuracy')  # Replace with your desired label
  plt.title('number of superpixels')  # Replace with your desired title
  plt.show()


def densifying(chro,prob):
    
    for idx in range (len(chro)):
       
            if chro[idx] == 0:
                if random.random() < prob:
                    chro[idx] = 1
    return chro,sum(chro)

def sparsing(chro,prob):
    
    for idx in range (len(chro)):
       
            if chro[idx] == 1:
                if random.random() < prob:
                    chro[idx] = 0
    return chro,sum(chro)



def deep_copy_struct(initial_struct):
    # Create a new struct with deepcopy for each attribute
    return type(initial_struct)(
        size=copy.deepcopy(initial_struct.size),
        acc=copy.deepcopy(initial_struct.acc),
        chro=copy.deepcopy(initial_struct.chro),
        pred=copy.deepcopy(initial_struct.pred),
        Fit=copy.deepcopy(initial_struct.Fit),
        rank=copy.deepcopy(initial_struct.rank),
        DominationSet=copy.deepcopy(initial_struct.DominationSet),
        DominatedCount=copy.deepcopy(initial_struct.DominatedCount),
        CrowdingDistance=copy.deepcopy(initial_struct.CrowdingDistance),
        im=copy.deepcopy(initial_struct.im)
    )

#############  HEURISTIC SEARCH
from ypstruct import struct

WIN=[]
Figure=5
imageLabel=y1_test[Figure]  
image=X1_test[Figure]
class_to_explain=np.where(y1_test[Figure]==1)
superpixels= slic(image, n_segments=5)   
nVar=np.unique(superpixels).shape[0]
el=0
TEMP=0
arTEMP=[]
limit=6

heuristic = struct(size=None, acc=None, chro=None, pred=None, Fit=[None,None], rank=None, DominationSet=None, DominatedCount=None, CrowdingDistance=None, im=None)

n = nVar
t=[None]*2**n
lst= [None] * 2**n

for i in range(2**n):
    t[i]=[str(x) for x in bin(i)[2:].zfill(n)]
    t[i]=(np.array(list(t[i]), dtype=int))
    
   
for i in range (2**n):
        active_pixels = np.where(t[i] == 1)[0]    
        mask=np.zeros(superpixels.shape)
        for active in active_pixels:
                mask[superpixels == active] = 1
        perturbed_image = copy.deepcopy(image)
        perturbed_image = perturbed_image*mask[:,:,np.newaxis]
        pic=np.expand_dims(perturbed_image, axis=0)
        pred = model.predict(pic)[0][class_to_explain]
        max_value=max(model.predict(pic)[0])
        k=model.predict(pic)[0]
        u=k.tolist()
        uu=u.index(max_value)      
               
        
        if len(active_pixels) == 0 :
          fit = (0.7*pred[0])
        
        if len(active_pixels) > 0:
          fit=(0.8*pred[0])+(0.2*((nVar-len(active_pixels)+1)/nVar))
         
          
        if   fit>TEMP and uu == class_to_explain[0][0]: 
          
                 lst[el]=perturbed_image
                 plt.imshow(perturbed_image)
                 plt.axis("off")
                 plt.show()
                 el=el+1
                 TEMP=fit
                 w=t[i]
                 fit
                 print(t[i],fit)
                 WIN.append(t[i])
                 # print(TEMP)
                 arTEMP.append(TEMP)

############################################
dis=[]
population=[]
paretoSize=[]
best_distance = float('inf') 
distance=best_distance
best_pareto_size = 0
import time
start=time.time()
TEMP=0
imageLabel=y1_test[Figure]  
image=X1_test[Figure]
class_to_explain=np.where(y1_test[Figure]==1)
superpixels= slic(image, n_segments=50)   
nVar=np.unique(superpixels).shape[0]
 
### show segmentized image
segmented_image = mark_boundaries(image, superpixels)
plt.imshow(segmented_image)

nVar=np.unique(superpixels).shape[0]
precisionP=[]
precisionN=[]
precision=[]
pareto=[]
NFE=len(t)

###### initial parent generation
from ypstruct import struct
FIRST=[None,None]
LAST=[None,None]
nPop=35
pc=0.9;
pm=0.4;
nc=2*round(pc*nPop/2)
nm=round(pm*nPop)
initial_solutions = struct(size=None, acc=None, chro=None, pred=None, Fit=[None,None], rank=None, DominationSet=None, DominatedCount=None, CrowdingDistance=None, im=None)
initial_pop = initial_solutions.repeat(nPop)
best_population = initial_solutions.repeat(nPop)
tagcheck=1
tagflag=1
phi=0.5

import copy

sh=1
for c in range(nPop):
       chromosome= np.random.binomial(1,phi, size=nVar)
       active_pixels = np.where(chromosome == 1)[0]
       print(active_pixels)
       mask=np.zeros(superpixels.shape)
       for active in active_pixels:
               mask[superpixels == active] = 1
       perturbed_image = copy.deepcopy(image)
       perturbed_image = perturbed_image*mask[:,:,np.newaxis]
       pic=np.expand_dims(perturbed_image, axis=0)
       pred = model.predict(pic)[0][class_to_explain]
       print(pred)
       plt.imshow(perturbed_image) 
       [initial_pop[c].im]=[perturbed_image]      
       [initial_pop[c].size]=[len(active_pixels)]
       [initial_pop[c].chro]=[chromosome]
       
       max_value=max(model.predict(pic)[0])
       k=model.predict(pic)[0]
       u=k.tolist()
       uu=u.index(max_value)
       S=len(active_pixels)        
       
       if uu == class_to_explain[0][0]:
           ac=1
           
       else:
           ac=0
       
       [initial_pop[c].acc] = [ac]
       [initial_pop[c].pred] = [pred]
       if initial_pop[c].size!=0:
            initial_pop[c].Fit= [(nVar-len(active_pixels)+1)/nVar,pred]
       else:
            initial_pop[c].Fit= [0,0]
       
       
       NFE=NFE+1

##########  NonDominatedSorting
[initial_pop, F, FrontsList]=NonDominatedSorting(initial_pop)
FrontsList=FrontsList[~FrontsList.isnull().any(axis=1)]


##########  CrowdingDistance

initial_pop=CalCrowdingDistance(initial_pop, FrontsList)



[initial_pop,ParetoFront,MaximumAcc]=SortPopulation(initial_pop)
MaximumAccuracy=MaximumAcc
distance=model.predict(X1_test)[Figure][class_to_explain][0]/ MaximumAcc
dis.append(distance)
best_population=initial_pop
paretoSize.append(len(ParetoFront))
print("distance", distance, 'tagcheck', tagcheck)
print ("Iteration",  "  ", 0,"      ","Number of F0 members", "  ",   len(ParetoFront)  )
pareto.append(len(ParetoFront))
precision.append(MaximumAccuracy)


import random
############### NSGA-II main loop
MaxIt=100
probability=0.5
alpha=0.2
for es in range (MaxIt):
   
   popc1=initial_solutions.repeat(int(nc/2))
   popc2=initial_solutions.repeat(int(nc/2)) 
   Xover=list(zip(popc1,popc2))
   for k in range (int(nc/2)):
       
       
       # Select First Parent
       # i1=RouletteWheelSelection(P)
       i1=BinaryTournamentSelection(initial_pop)
       # i1=random.randint(0,nPop-1)
       p1=initial_pop[i1].chro
       # Select Second Parent
       # i2=RouletteWheelSelection(P)
       i2=BinaryTournamentSelection(initial_pop)
       # i2=random.randint(0,nPop-1)
       p2=initial_pop[i2].chro
       #Apply Crossover
       Xover[k][0].chro,Xover[k][1].chro=np.array(SinglePointCrossover(p1,p2))
       #Evaluate Offspring
       Xover[k][0].pred,Xover[k][0].im,Xover[k][0].acc,Xover[k][0].Fit=perturb_image(image,Xover[k][0].chro,superpixels,es)
       Xover[k][0].size=Xover[k][0].chro.sum(0)
       
       Xover[k][1].pred,Xover[k][1].im,Xover[k][1].acc, Xover[k][1].Fit=perturb_image(image,Xover[k][1].chro,superpixels,es)
       Xover[k][1].size=Xover[k][1].chro.sum(0)
       popc=initial_solutions.repeat(nc)
   i=0
   for s in range (len(Xover)):
       for j in range(2):
            popc[i]=Xover[s][j]
            i=i+1
    
###### mutation


   popm=initial_solutions.repeat(nm)
   

###########Adaptive Bit Flip mutation     
   
   if distance>3 and tagcheck==10:
               
                   probability= probability + alpha *(1- probability) 
                   print(probability)
                   print('Densifying')
                   print("distance", distance, 'tagcheck', tagcheck)
                   for k in range(nm):
   
                        i=random.randint(0,nPop-1)
                        p=initial_pop[i].chro
                        popm[k].chro,popm[k].size = densifying(p, probability)
                        popm[k].pred,popm[k].im,popm[k].acc,popm[k].Fit=perturb_image(image,popm[k].chro,superpixels,es)
                
                   tagcheck=0
               
   elif distance>3 and tagcheck<10: 
                print("distance", distance, 'tagcheck', tagcheck)

                # tagcheck=tagcheck+1 
                for k in range(nm):
   
                    i=random.randint(0,nPop-1)
                    p=initial_pop[i].chro
                    popm[k].chro,popm[k].size=Mutate2(p)
                    popm[k].pred,popm[k].im,popm[k].acc,popm[k].Fit=perturb_image(image,popm[k].chro,superpixels,es)
   elif distance<=3  and tagflag==10:
        print("distance", distance, 'tagflag', tagflag)

        probability = probability - alpha * probability
        # probability = 0.7
        print(probability)
        print('Sparsing')

        for k in range(nm):

            i=random.randint(0,nPop-1)
            p=initial_pop[i].chro
            popm[k].chro,popm[k].size = sparsing(p, probability)
            popm[k].pred,popm[k].im,popm[k].acc,popm[k].Fit=perturb_image(image,popm[k].chro,superpixels,es)
        
        tagflag=0
      
       
   elif distance<=3 and tagflag<10:
         print("distance", distance, 'tagflag', tagflag)
      
      
         for k in range(nm):

              i=random.randint(0,nPop-1)
              p=initial_pop[i].chro

              popm[k].chro,popm[k].size=Mutate2(p)

              popm[k].pred,popm[k].im,popm[k].acc,popm[k].Fit=perturb_image(image,popm[k].chro,superpixels,es)
       
       
#######  merge population        
   initial_pop= initial_pop+popc+popm    
######  SORT

####### FrontsList contain the fronts' members including paretofront and the rest
   [initial_pop, F, FrontsList]=NonDominatedSorting(initial_pop)
   FrontsList=FrontsList[~FrontsList.isnull().any(axis=1)]
########Crowding Distance Calculation#################  
   initial_pop=CalCrowdingDistance(initial_pop, FrontsList)
#########Sort population based on rank and crowding distance#################     
   [initial_pop, ParetoFront, MaximumAcc]=SortPopulation(initial_pop)
####Truncate
   initial_pop=initial_pop[0:nPop]
   # print("len initial_pop", " ", len(initial_pop))

   FF=[]
   M=[]
   
   for dd in range (len(initial_pop)):
         if (initial_pop[dd].rank==0) :
            FF.append(initial_pop[dd].Fit)
            M.append(initial_pop[dd].pred)
   if len(M)!=0:
        MaximumAcc=max(M)
   # print("Len pareto front", " ", len(FF))
    
   distance=model.predict(X1_test)[Figure][class_to_explain][0]/MaximumAcc
   paretoSize.append(len(FF))

   
   if  distance> 3 and abs(distance-dis[-1])==0:
       tagcheck=tagcheck+1
       tagflag=1
       
   if distance> 3 and abs(distance-dis[-1])!=0:
       tagcheck=1
       tagflag=1
   if  distance <= 3 and abs(distance-dis[-1])==0:
      tagflag=tagflag+1
      tagcheck=1
   if  distance <= 3 and abs(distance-dis[-1])!=0:
       tagflag=1
       tagcheck=1
   dis.append(distance)
   if (distance < best_distance) or (distance == best_distance and len(FF) > best_pareto_size):
        best_distance = distance
        best_population = [deep_copy_struct(sol) for sol in initial_pop]

        best_pareto_size = (len(FF))
   pareto.append(len(FF))
   

   print ("Iteration",  "  ", es+1,"      ","Number of F0 members", "  ",  len(FF)   )
end=time.time()
print("Time = ", end-start,  "NFE = ", NFE)







##### remove those solutions in pareto front with of initial_pop that the  acc=0
new_initial_pop = [solution for solution in best_population if solution.rank == 0 and solution.acc == 1]
# new_initial_pop = [solution for solution in best_population if solution.rank == 0  ]

# new_initial_pop=initial_pop
######  remove identical solutions in ParetoFront
ip = struct(size=None, acc=None, pred=None, Fit=[None,None], rank=None, DominationSet=None, DominatedCount=None, CrowdingDistance=None, im=None)
unique_solutions = ip.repeat(len(new_initial_pop))
templist=[]

# Iterate through 'initial_pop' and filter duplicates based on fitness vectors
for c in range (len(new_initial_pop)):
    fitness_vector = new_initial_pop[c].Fit  # Convert the fitness vector to a tuple
    
    # Check if the fitness vector is unique
    if fitness_vector not in templist:
        print(c)
        unique_solutions[c] = new_initial_pop[c]
        templist.append(fitness_vector)
t1=[]
cc=0
for dd in range (len(unique_solutions)):
    if (unique_solutions[dd].Fit != [None,None]):
        plt.imshow(unique_solutions[dd].im)
        plt.title(dd+1)
        plt.axis('off')
        plt.show()
        t1.append(unique_solutions[dd].im)
        cc=cc+1
        print(unique_solutions[dd].Fit)

######## In case you want to see the optimal images individually
# plt.imshow(t1[0])
# plt.axis("off")







#### VOTING
mask = np.any(t1[0] != [0, 0, 0], axis=-1)
pixel_counts = np.zeros_like(mask, dtype=np.uint16)
for img in t1:
    pixel_mask = np.any(img != [0, 0, 0], axis=-1)
    pixel_counts += pixel_mask
final_mask = pixel_counts >= (len(t1)//2+1)
# final_mask = pixel_counts == (len(t1))
# 

final_mask_3d = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.float32)
final_mask_3d[final_mask] = 1
plt.imshow(image*final_mask_3d)
plt.axis('off')
#########   In case you want to try the UNION of optimal images instead of majority voting

union_mask = np.zeros(t1[0].shape[:2], dtype=bool)
# Combine masks by taking the union
for img in t1:
    pixel_mask = np.any(img != [0, 0, 0], axis=-1)
    union_mask = np.logical_or(union_mask, pixel_mask)

# Convert the union mask to a 3D representation for visualization
union_mask_3d = np.zeros((*union_mask.shape, 3), dtype=np.float32)
union_mask_3d[union_mask] = 1

# Assuming 'image' is the original image you want to overlay the mask on
plt.imshow(image * union_mask_3d)
plt.axis('off')
plt.show()
final_mask_3d=union_mask_3d




#####CREATING BASE image
# BASE= np.all(WIN, axis=0).astype(int)
BASE=w
active_pixels = np.where(BASE == 1)[0]
print(active_pixels)
superpixels= slic(image, n_segments=5)   
mask=np.zeros(superpixels.shape)
for active in active_pixels:
        mask[superpixels == active] = 1
perturbed_image = copy.deepcopy(image)
perturbed_image = perturbed_image*mask[:,:,np.newaxis]
plt.imshow(perturbed_image)
plt.axis("off")

######### Generating final  image with BLACK background
black_back=np.where(perturbed_image != 0, image*final_mask_3d, 0)
plt.imshow(black_back)
plt.axis('off')

######### Generating final  image with WHITE background
plt.imshow(np.where(black_back == 0, 0.5, black_back))
plt.axis('off')




#################### Plot distance########################
plt.plot(dis)
plt.xlabel('Iteration')
plt.ylabel('Accuracy deviation from the original image')
plt.grid(True)
plt.title('Healthy')  
plt.ylim(0.9999,1.001)

#################### PLOT performance graph

x = [i for i in range(1, 102)]

# Plot the points
plt.scatter(x, pareto, color='red', label='Points')

# Connect the points with a line
plt.plot(x, pareto, color='blue', linestyle='-', linewidth=2, label='Line')

# Set labels and title
plt.xlabel('Iteration')
plt.ylabel('Number of Pareto front members')
plt.grid(True)
plt.title('Canker')      



############# Calculate Numeric accuracy of explanation
worst=worst55.flatten()
delineation=image29_delineated.flatten()
calculated=black_back.flatten()


distance =0
for i in range(len(delineation)):
        # distance += np.square(input_image[i]-con[i])
        # distance += np.square(input_image[i]-maj[i])
        distance += np.square(delineation[i]-calculated[i])
        
np.sqrt(distance)
##########################################

 

        
        


############ LIME  implementation

import lime
from lime import lime_image

start=time.time()
explainer=lime_image.LimeImageExplainer()
explanation=explainer.explain_instance(X1_test[5],model.predict, num_samples=4643,segmentation_fn=slic)
image,mask=explanation.get_image_and_mask(model.predict(X1_test[5].reshape((1,256,256,3))).argmax(axis=1)[0], negative_only=False, positive_only=False, num_features=50)
end=time.time()
print("time = ", end-start)
plt.imshow(mark_boundaries((image),mask))
plt.axis('off')










######## NFE  Evaluation of EGAE and MOGAE
Be = [4643, 4651, 4651, 4651]
Af = [3400, 4108, 2025, 2962]
Aff= [4743,5972,3393,4728]

# barWidth = 0.3
# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
 


w=0.5*barWidth
# plt.bar(bar_positions, Af, color='lightblue', width=barWidth, label='EGAE-I')
# Make the plot

plt.bar(br1, Af, color='lightblue', width=w, label='EGAE-I')
plt.bar([x + 0.5 * barWidth for x in br1], Aff, color='blue', width=w, label='EGAE-II')
plt.bar([x + barWidth for x in br1], Be, color='red', width=w, label='MOGAE')

# Adding Xticks
plt.ylabel('NFE')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Black spot', 'Canker', 'Greening', 'Healthy'])
 
plt.legend(loc='lower center', ncol=3, bbox_to_anchor=[0.5, -0.2])
# plt.title('Colon')
plt.show()



###################  Generating the figure of the impact of delta on Omega
yii=np.zeros(20)
yii[0]=1


probability = 0.5

alpha = 0.2
for i in range(1, 20):
    probability = probability - (alpha * probability)
    print(probability)  
    yii[i]=probability
   

    
x = [i for i in range(1, 21)]    

plt.scatter(x, Di, color='orange', marker='s',s=100)
plt.plot(x, Di, color='black', linestyle='-', linewidth=2)

plt.scatter(x, Dii, color='orange', marker='s')
plt.plot(x, Dii, color='black', linestyle='--', linewidth=2)

plt.scatter(x, yi, color='red', s=100)
plt.plot(x, yi, color='blue', linestyle='-', linewidth=2)

plt.scatter(x, yii, color='red')
plt.plot(x, yii, color='blue', linestyle='--', linewidth=2)
plt.grid(True)
plt.xticks(range(1, 21, 1))
plt.legend(['Densifying (δ = 0.4)', 'Densifying (δ = 0.2)', 'Sparsing (δ = 0.4)', 'Sparsing (δ = 0.8)' ], loc = "best")
plt.xlabel('Iteration')
plt.ylabel('Ω')
plt.show()   

 