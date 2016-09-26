import csv
import numpy as np 
import matplotlib.pyplot as plt

'''Coding logical regression '''
    
''' load the dataset '''
def load_csv():
    
    lines = csv.reader(open('Project1_data.csv','rb'))
    dataset = list(lines)
    
    newset = []

    for i in range(1,len(dataset)):
        temp = []
        for z in range(len(dataset[i])):
            for char in dataset[i][z]:
                if char in "!.?/\;- @#$%^&*()":
                    dataset[i][z] = dataset[i][z].replace(char,'')
            temp.append(str(dataset[i][z]).lower())
        newset.append(temp)
        #include list comprehension for removing special characters and space
    return newset

''' make the features '''    
def make_features(dataset):
    
    x = []
    y = []
    y_test = []
    f = open('2016.txt', 'w')
    
    for k in range(len(dataset)):

     y_past = 0 #will return 1 if the person did the full mtl marathon in 2014
     y_2015 = 0
     
     events_in_mtl = 0 # num of events in mtl == x1
     events_not_mtl = 0 # events not in mtl== x2
     
     x1 = 0 #render 1 if events_in_mtl >= 2	
     x2 = 0 #render 1 if events_not_mtl >= 2
     
     full_oasis_compl = 0 #number of completed full marathons in mtl == x3
     half_oasis_compl = 0 #number of completed half marathons in mtl == x4
     
     x3 = 0 #render 1 if full_oasis_compl >= 1
     x4 = 0 #render 1 if half_oasis_compl >= 2
     
     length = len(dataset[k])
     number_races = (length-1)/5 #total number of races that someone could have done
     
     for i in range(number_races): #loops through all possible races for i
	    
         event_date = dataset[k][5*i+1]
         event_name = dataset[k][5*i+2]
         event_type = dataset[k][5*i+3]
         event_time = dataset[k][5*i+4] 
         
        #if they didn't finish the race -1 (now 1) go to next race        
         if dataset[k][5*i+4] == '1':
             continue 

        #eliminate values from 2016
         if '2016' in dataset[k][5*i+1]:
            continue

         if '2015' in dataset[k][5*i+1]:
            if ('demi' or 'half' or '21') not in event_type:
			if (('marathon' or '42') in event_type) and ('oasis' in event_name):
		    		y_2015 = 1
         
        #if someone ever did the FULL Oasis Montreal Marathon in the past
         if ('2014' or '2013'or '2012') in event_date:
		if ('demi' or 'half' or '21') not in event_type:
			if (('marathon' or '42') in event_type) and ('oasis' in event_name):
		    		y_past = 1
        
        #looks at events in mtl
         if 'montreal' in event_name:
              events_in_mtl += 1
               #looks at half-marathons in montreal
              if (('demi' or 'half' or '21') in event_type) and ('oasis' in event_name): 		
			if event_time != '1': 	 #checks to see if they are completed or not
				half_oasis_compl += 1
              #looks at full marathons in montreal
              elif (('marathon' or '42') in event_type) and ('oasis' in event_name): 
			if event_time != '1':		#checks to see if completed or not
				full_oasis_compl += 1	
    
         #looks at events outside mtl
         if (('toronto' or 'ottawa' or 'sherbrooke') in event_name) or ('montreal' not in event_name): 
		events_not_mtl += 1
  
         if events_in_mtl >= 2: 
		x1 = 1
         if events_not_mtl >= 2: 
		x2 = 1
         if full_oasis_compl >= 1: 
		x3 = 1
         if half_oasis_compl >= 2: 
		x4 = 1
  
         #get the training data 
         x.append([x1, x2, x3, x4])
         y.append(y_past)
         
         #validation set 
         y_test.append(y_2015)
         
         #write out stuff to a file 
         f.write("".join(str(y_2015)))
         f.write('\n')
                
    #training set
    x_new = np.array(x) #1xn
    y_new = np.array(y)
       
    #omega = w is mX1
    shape = x_new.shape[1]
    omega = np.zeros(shape) 
    
    #get weight
    weight = gradient(omega,x_new,y_new)
    print weight 
    
    #predict the runners that will run this year     
    predicted_y = predict(weight, x_new)
    
    #prediction accuracy = predicted_correctly/total 
    run = np.sum(y_test == predicted_y)
    print (float) (run)/len(x_new) * 100
    
''' logistic funtion ''' 
def sigmoid(omega,x):
  
   return (float)(1) / (1+np.e**(-x.dot(omega)))

''' calculate logistic value '''
def log_gradient(omega, x, y):
   
   for i in range(len(x)):  
     first = sigmoid(omega, x[i]) - y[i]
     final= first.T*(x[i])
    
   return final

''' compute the cost of regression '''
def compute_cost(omega,x,y):
    
    omegaX = sigmoid(omega,x)
    
    j1 = y * np.log(omegaX) #this is logistic regression equation split into 2 
    j2 = (1-y) * np.log(1 - omegaX)
    err = -j1-j2
    
    return np.mean(err)
    
''' gradient descent '''
def gradient(omega_k,x,y,l_value=0.001,converge=0.001):
    
    cost_arr = [] #keep track of weight values 
    cost = compute_cost(omega_k, x, y)
    cost_arr.append([0, cost])
    
    delta = 1 #step value 
    j = 1
    
    #iterate till there is no more change 
    while(delta>converge):
        old_cost = cost
        omega_k = omega_k - (l_value*log_gradient(omega_k,x,y))
        cost = compute_cost(omega_k,x,y)
        cost_arr.append([j, cost])
        delta = old_cost - cost 
        j+=1

    ''' learning curve '''
    plt.plot(cost_arr[0], cost_arr[1])
    plt.ylabel("Cost")
    plt.xlabel("Iteration")
    plt.show()
        
    return omega_k #return weight values 
    
''' predict y=0 or y=1 for each runner '''
def predict(omega,x,hard=True):
    
    prediction = sigmoid(omega,x)
    prediction_value = np.where(prediction >= .5, 1, 0)
    
    if hard:
        return prediction_value
        
    return prediction 
 
''' main '''
def main():
    
    make_features(load_csv())

main()

