import numpy as np
import copy

class RANN:
    def __init__(self,
                 layer_list,
                 weight_scaler = 0.05,
                 firing_rate_scaler = 0.1,
                 learning_rate = 0.01,
                 loss_function = 'mse',
                 optimizer = 'sgd'):
        
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.shape = layer_list #[x,20,10,1]
        self.loss_func = loss_function
        n = len(layer_list)

        self.layers = []
        for i in range(n):
            self.layers.append(np.zeros(self.shape[i]))

        self.rate = []
        for i in range(n):
            self.rate.append(np.zeros((self.shape[i],1)))
        self.rate[-1] = firing_rate_scaler * np.ones((self.rate[-1].shape))
        
        self.wplus = []
        self.wminus = []
        for i in range(n-1):
            self.wplus.append(np.zeros((self.layers[i].size,
                                          self.layers[i+1].size)))
            self.wminus.append(np.zeros((self.layers[i].size,
                                          self.layers[i+1].size)))
        
        self.Q = []
        for i in range(n):
            self.Q.append(np.zeros((self.shape[i],1)))
            
        self.lambda_plus = []
        self.lambda_minus = []    
        
        self.D = []
        self.global_var2 = []
        self.global_var = []
        
        self.pre_grad_p = copy.deepcopy(self.wplus)
        self.pre_grad_m = copy.deepcopy(self.wminus)
        self.pre_grad_pm = np.array([copy.deepcopy(self.wplus),copy.deepcopy(self.wminus)])
        self.pre_grad_pm_2 = copy.deepcopy(self.pre_grad_pm)
        self.pre_grad_pm_3 = copy.deepcopy(self.pre_grad_pm)
        self.init_weights(weight_scaler)
        
        self.bp_counter=0

    def init_weights(self,weight_scaler):
        
        for i in range(len(self.wplus)):
            self.wplus[i] = weight_scaler*np.random.uniform(size=(self.wplus[i].shape[0],self.wplus[i].shape[1]))
        for i in range(len(self.wminus)):
            self.wminus[i] = weight_scaler*np.random.uniform(size=(self.wminus[i].shape[0],self.wminus[i].shape[1]))
        return 1
    
    def calculate_rate(self):
        for layer_num in range(len(self.layers)-1): #there is no rate for output layer
            for neuron in range(len(self.layers[layer_num])):
                self.rate[layer_num][neuron] = (self.wplus[layer_num][neuron] + self.wminus[layer_num][neuron]).sum()

        return 1

    
    def feedforward(self,input_list):
        self.D.clear() #clear list of D matrixes for the next iteration
        self.lambda_plus = np.where(input_list > 0, input_list, 0).reshape(-1,1)
        self.lambda_minus = np.where(input_list < 0, -input_list, 0).reshape(-1,1)
        
        # Input Layer
        self.D.append(self.rate[0]+self.lambda_minus)
                
        self.Q[0]=self.lambda_plus/self.D[0]
        np.clip(self.Q[0],0,1,out=self.Q[0])

        for i in range(1,len(self.shape)):
            # Hidden Layer
            T_plus = np.dot(self.wplus[i-1].transpose(),self.Q[i-1])
            T_minus =  np.dot(self.wminus[i-1].transpose(),self.Q[i-1])
            self.D.append(self.rate[i] + T_minus)
            self.Q[i] = T_plus / self.D[i]
            np.clip(self.Q[i],0,1,out=self.Q[i])

        return copy.deepcopy(self.Q[-1])
    
    def backpropagation(self,real_output,tmp1=np.zeros((2,1))):
        self.bp_counter=self.bp_counter+1
        if self.optimizer=='nag':
            weights_pm = np.array([self.wplus,self.wminus])
            x_ahead = weights_pm-self.pre_grad_pm*0.9
            self.wplus,self.wminus=copy.deepcopy(list(x_ahead[0])),copy.deepcopy(list(x_ahead[1]))
            
            for i in range(len(self.wminus)):
                np.clip(self.wminus[i],a_min=0.001,a_max=None)
                np.clip(self.wplus[i],a_min=0.001,a_max=None)
            
        loss = 0
        grad_p = []
        grad_m = []
        for i in reversed(range(len(self.shape)-1)):
            ver_grad = ((self.wplus[i].transpose()-(self.wminus[i].transpose()*self.Q[i+1]))/self.D[i+1]).transpose()
            do_dwp1 = ((self.Q[i] @ (1/self.D[i+1]).transpose() + (-self.Q[i]/self.D[i])*ver_grad).transpose()*tmp1).transpose()
            do_dwm1 = ((-self.Q[i] @ (self.Q[i+1]/self.D[i+1]).transpose() + (-self.Q[i]/self.D[i])*ver_grad).transpose()*tmp1).transpose()
            tmp1 = ver_grad @ tmp1
            grad_p.append(do_dwp1)
            grad_m.append(do_dwm1)
        grad_p.reverse()
        grad_m.reverse()
        
        grad_p = np.asarray(grad_p)
        grad_m = np.asarray(grad_m)
        grad_pm = np.array([grad_p,grad_m])
        
        if self.optimizer == 'sgd':
            self.update_weigths(grad_pm*self.learning_rate)
            #self.update_weigths([grad_p*self.learning_rate,grad_m*self.learning_rate])

        elif self.optimizer == 'momentum':
            moment_pm = self.pre_grad_pm*0.9;
            self.pre_grad_pm = moment_pm + grad_pm*self.learning_rate
            self.update_weigths(self.pre_grad_pm)
            #self.pre_grad_pm=copy.deepcopy(grad_pm)
        
        elif self.optimizer == 'nag':

            self.pre_grad_pm = self.pre_grad_pm*0.9 + grad_pm*self.learning_rate
            self.update_weigths(self.pre_grad_pm)
            
            #self.pre_grad_pm = self.pre_grad_pm*0.9 + grad_pm#*self.learning_rate
            #grads = self.pre_grad_pm*0.9 + grad_pm#*self.learning_rate
            #self.update_weigths(grads*self.learning_rate)
            #self.gradient_list=copy.deepcopy(grads)
            
        elif self.optimizer == 'adagrad':
            grad_pm_2 = np.square(grad_pm)
            self.pre_grad_pm+=copy.deepcopy(grad_pm_2)
            for i in range(grad_pm.shape[1]):
                grad_pm[0,i] = (self.learning_rate/(np.sqrt(self.pre_grad_pm[0,i]+0.000001)))*grad_pm[0,i]
                grad_pm[1,i] = (self.learning_rate/(np.sqrt(self.pre_grad_pm[1,i]+0.000001)))*grad_pm[1,i]
            self.update_weigths(grad_pm)
            #self.pre_grad_pm+=copy.deepcopy(grad_pm)
            
        elif self.optimizer == 'adadelta': #gradient_list_2->gt//gradient_list->teta_t
            eps=0.000001;beta=0.90;
            grad_pm_2 = np.square(grad_pm)
            
            self.pre_grad_pm_2 = beta*self.pre_grad_pm_2 + (1-beta)*grad_pm_2
            
            delta_teta = copy.deepcopy(self.pre_grad_pm)
            for i in range(grad_pm.shape[1]):
                #delta_teta[0,i] = (self.learning_rate/(np.sqrt(self.pre_grad_pm_2[0,i]+0.000001)))*grad_pm[0,i]
                #delta_teta[1,i] = (self.learning_rate/(np.sqrt(self.pre_grad_pm_2[1,i]+0.000001)))*grad_pm[1,i]
                delta_teta[0,i] = (np.sqrt(self.pre_grad_pm[0,i]+eps)/(np.sqrt(self.pre_grad_pm_2[0,i]+0.000001)))*grad_pm[0,i]
                delta_teta[1,i] = (np.sqrt(self.pre_grad_pm[0,i]+eps)/(np.sqrt(self.pre_grad_pm_2[1,i]+0.000001)))*grad_pm[1,i]

            self.pre_grad_pm = beta*self.pre_grad_pm + (1-beta)*np.square(delta_teta)

            for i in range(grad_pm.shape[1]):
                grad_pm[0,i] = (np.sqrt(self.pre_grad_pm[0,i]+eps)/(np.sqrt(self.pre_grad_pm_2[0,i]+0.000001)))*grad_pm[0,i]
                grad_pm[1,i] = (np.sqrt(self.pre_grad_pm[1,i]+eps)/(np.sqrt(self.pre_grad_pm_2[1,i]+0.000001)))*grad_pm[1,i]
  
            self.update_weigths(grad_pm)
            #self.pre_grad_pm = beta*self.pre_grad_pm + (1-beta)*grad_pm_2
            
        elif self.optimizer == 'rmsprop':
            eps=0.00000001
            grad_pm_2 = np.square(grad_pm)
            self.pre_grad_pm_2 = 0.9*self.pre_grad_pm_2 + 0.1*grad_pm_2

            for i in range(grad_pm.shape[1]):
                grad_pm[0,i] = (self.learning_rate/(np.sqrt(self.pre_grad_pm_2[0,i]+eps)))*grad_pm[0,i]
                grad_pm[1,i] = (self.learning_rate/(np.sqrt(self.pre_grad_pm_2[1,i]+eps)))*grad_pm[1,i]
         
            self.update_weigths(grad_pm)
            
        elif self.optimizer == 'adam':
            eps=0.000001;beta1=0.9;beta2=0.999;
            grad_pm_2 = np.square(grad_pm)
            self.pre_grad_pm = beta1*self.pre_grad_pm + (1-beta1)*grad_pm
            self.pre_grad_pm_2 = beta2*self.pre_grad_pm_2 + (1-beta2)*grad_pm_2
            gt = self.pre_grad_pm/(1-beta1**self.bp_counter)
            gt2 = self.pre_grad_pm_2/(1-beta2**self.bp_counter)
            
            for i in range(grad_pm.shape[1]):
                grad_pm[0,i] = (self.learning_rate/(np.sqrt(gt2[0,i])+eps))*gt[0,i]
                grad_pm[1,i] = (self.learning_rate/(np.sqrt(gt2[1,i])+eps))*gt[1,i]
         
            self.update_weigths(grad_pm)

        elif self.optimizer == 'adamax':
            eps=0.000001;beta1=0.9;beta2=0.999;
            
            grad_pm_2 = np.square(grad_pm)
            self.pre_grad_pm = beta1*self.pre_grad_pm + (1-beta1)*grad_pm
            gt = self.pre_grad_pm/(1-beta1**self.bp_counter)

            for i in range(self.pre_grad_pm_2.shape[1]):
                if self.bp_counter==1:
                    self.pre_grad_pm_2[0,i].fill(eps)
                    self.pre_grad_pm_2[1,i].fill(eps)
                self.pre_grad_pm_2[0,i] =np.maximum(beta2*self.pre_grad_pm_2[0,i],abs(grad_pm)[0,i])
                self.pre_grad_pm_2[1,i] =np.maximum(beta2*self.pre_grad_pm_2[1,i],abs(grad_pm)[1,i])

            #learning rate is 0.002 as an adviced value
            grad_pm = (self.learning_rate / (self.pre_grad_pm_2+eps)) * gt
                       
                       
            self.update_weigths(grad_pm)

        elif self.optimizer == 'nadam':
            eps=0.000001;beta1=0.9;beta2=0.999;
            
            grad_pm_2 = np.square(grad_pm)
            self.pre_grad_pm = beta1*self.pre_grad_pm + (1-beta1)*grad_pm
            self.pre_grad_pm_2 = beta2*self.pre_grad_pm_2 + (1-beta2)*grad_pm_2
            gt = self.pre_grad_pm/(1-beta1**self.bp_counter)
            gt2 = self.pre_grad_pm_2/(1-beta2**self.bp_counter)
            gt = (beta1*gt-((1-beta1)/(1-beta1**self.bp_counter))*gt)
            for i in range(grad_pm.shape[1]):
                grad_pm[0,i] = (self.learning_rate/(np.sqrt(gt2[0,i])+eps))*gt[0,i]
                grad_pm[1,i] = (self.learning_rate/(np.sqrt(gt2[1,i])+eps))*gt[1,i]
            self.update_weigths(grad_pm)
        
        elif self.optimizer == 'amsgrad':
            eps=0.000001;beta1=0.9;beta2=0.999;
            
            grad_pm_2 = np.square(grad_pm)
            self.pre_grad_pm = beta1*self.pre_grad_pm + (1-beta1)*grad_pm
            self.pre_grad_pm_2 = beta2*self.pre_grad_pm_2 + (1-beta2)*grad_pm_2
            gt = self.pre_grad_pm/(1-beta1**self.bp_counter)
            gt2 = self.pre_grad_pm_2/(1-beta2**self.bp_counter)
            
            for i in range(self.pre_grad_pm_3.shape[1]):
                self.pre_grad_pm_3[0,i] =np.maximum(self.pre_grad_pm_3[0,i],gt2[0,i])
                self.pre_grad_pm_3[1,i] =np.maximum(self.pre_grad_pm_3[1,i],gt2[1,i])
           
            for i in range(grad_pm.shape[1]):
                grad_pm[0,i] = (self.learning_rate/(np.sqrt(self.pre_grad_pm_3[0,i])+eps))*gt[0,i]
                grad_pm[1,i] = (self.learning_rate/(np.sqrt(self.pre_grad_pm_3[1,i])+eps))*gt[1,i]
            self.update_weigths(grad_pm)
        else:
            raise Exception('Unknown optimizer : \'{}\''.format(self.optimizer))
        
        return loss

    def update_weigths(self,grad_list):
        
        for i in range(len(self.shape)-1):
            self.wplus[i] = copy.deepcopy(np.clip(self.wplus[i] - grad_list[0][i],a_min=0.001,a_max=None))
            self.wminus[i] = copy.deepcopy(np.clip(self.wminus[i] - grad_list[1][i],a_min=0.001,a_max=None))

    def softmax(self,xs):
      return np.exp(xs) / sum(np.exp(xs))
