import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np
import matplotlib.pyplot as plt

def cross(a,b):
    """
    Compute cross product with batching. Currently only allows particular number of dimensions in the input...
    a - [!,3] tensor
    b - [*,!,3] tensor
    e.g.
    a.shape = [20,3]
    b.shape = [15,20,3]
    result.shape = [15,20,3]
    TODO: Generalise for any compatible input batch shapes
    """
    size = a.shape[0]
    A = tf.Variable([[tf.zeros(size), -a[:,2], a[:,1]],[a[:,2], tf.zeros(size), -a[:,0]],[-a[:,1], a[:,0],tf.zeros(size)]])
    A = tf.transpose(A,[2,0,1])
    A = A[None,:,:,:]
    b = b[:,:,:,None]
    return (A@b)[:,:,:,0]

class PathInference:
    def __init__(self,obstimes, observations, kernel, likenoisescale, Nind=None, Z=None, IndStartEnd=None, memlimit=1e6):
        """
        obstimes : The times of the N observations (a 1d numpy array)
        observations : The observations themselves. For a d-dimensional space, this will consist of
                       an [N x 2*d] numpy array.
        kernel : a method implementing the kernel
                    this should take two tensorflow tensors X1 and X2 (consisting of Ax2 and Bx2
                    arrays, the first column is the time, the second the index of the axis. For example:
                          [[0, 0]
                           [5, 0]
                           [0, 1]
                           [5, 1]
                           [0, 2]
                           [5, 2]] (for 3d data)
                    it should return an (A x B) tensorflow tensor of covariances.                           
    
        Inducing points can be either:
        (a) selected automatically.
        (b) selected automatically, using Nind inducing points.
        (c) selected automatically, using inducing points evenly spaced between the times in tuple IndStartEnd
        (d) selected manually (by setting Z).
        Nind : number of inducing points (default is 1+int(3*np.max(obstimes))).
        memlimit : maximum number of values in a numpy array, default = 1 million (which is about 8 to 16Mb of memory).
        """
        self.obstimes = obstimes
        self.observations = observations
        self.likenoisescale = likenoisescale
        self.dims = int(observations.shape[1]/2)
        self.memlimit = memlimit

        self.kernel = kernel
        if Nind is None:
            Nind = 1+int(3*(np.max(self.obstimes)-np.min(self.obstimes))/kernel.ls)
            #np.max(self.obstimes)-np.min(self.obstimes))
            #Nind = 1+int(3*np.max(obstimes))
            print("Using %d inducing points." % Nind)
        self.Nind = Nind
        
        if IndStartEnd is None:
            min_time = None
            max_time = None
        else:
            min_time = IndStartEnd[0]
            max_time = IndStartEnd[1]
        #Build the inducing point locations:        
        if Z is None:
            self.Z = self.buildinputmatrix(self.Nind,min_time,max_time)
        

    def compute_matrices(self,X,Z):
        """TODO: Rename as A and B not A and Z"""
        if X.shape[0]*Z.shape[0]>self.memlimit:
            raise Exception("May exceed memory avialable (X x Z is %d x %d)" % (X.shape[0],Z.shape[0]))
        
        Kzz = self.kernel.K(Z,Z)+np.eye(Z.shape[0],dtype=np.float32)*self.jitter
        Kxx = self.kernel.K(X,X)+np.eye(X.shape[0],dtype=np.float32)*self.jitter
        Kxz = self.kernel.K(X,Z)
        Kzx = tf.transpose(Kxz)
        KzzinvKzx = tf.linalg.solve(Kzz,Kzx)
        KxzKzzinv = tf.transpose(KzzinvKzx)
        KxzKzzinvKzx = Kxz @ KzzinvKzx
        return Kzz,Kxx,Kxz,Kzx,KzzinvKzx,KxzKzzinv,KxzKzzinvKzx
        
    
    def buildinputmatrix(self,times,min_time=None,max_time=None):
        """
        times can either be the number of times or a list of times.
        Constructs a matrix of [dims*size,2], where the first column is time and second is axis index.
        - min_time and max_time specify the linspace range of times (default to a time just before and
          after the obstimes).
        - size = number of points        
        Returns a [self.dims*size,2] matrix
        """
        if isinstance(times,int):
            size = times
            
            buffer_time = 0.1*(np.max(self.obstimes)-np.min(self.obstimes)) #10% of total time on either side
            if max_time is None:
                max_time = np.max(self.obstimes)+buffer_time
            if min_time is None:
                min_time = np.min(self.obstimes)-buffer_time
            
            A = []
            for ax in range(self.dims):
                Aax = np.c_[np.linspace(min_time,max_time,size),np.full(size,ax)]
                A.extend(Aax)
            A = np.array(A)
            A = tf.Variable(A,dtype=tf.float32)    
            return A
        else:
            times = np.array(times)
            assert len(times.shape)==1, "Expecting a 1d list of times, or the number of time points."
            A = []
            for ax in range(self.dims):
                Aax = np.c_[times,np.full(len(times),ax)]
                A.extend(Aax)
            A = np.array(A)
            A = tf.Variable(A,dtype=tf.float32)
            return A
    
    def getcov(self,scale):
        return tf.linalg.band_part(scale, -1, 0) @ tf.transpose(tf.linalg.band_part(scale, -1, 0))
    
    
    def getpredictions(self,Xs):
        """
        Returns a tensor (matrix) of means, and a tensor of covariances.
        """
        Kzz,Kxx,Kxz,Kzx,KzzinvKzx,KxzKzzinv,KxzKzzinvKzx = self.compute_matrices(Xs,self.Z)
        m = self.Z.shape[0]
        qf_mu = (KxzKzzinv @ self.mu)[:,0]
        qf_cov = Kxx - KxzKzzinvKzx + KxzKzzinv @ self.getcov(self.scale) @ KzzinvKzx
        size = int(Xs.shape[0]/self.dims) #number of input points
        C = tf.transpose(tf.concat([qf_cov[i::size,i::size][:,:,None] for i in range(size)],axis=2),[2,0,1])
        M = tf.transpose(tf.reshape(qf_mu,[self.dims,size]),[1,0])
        return M, C
    
    def tryrun(self, iterations=500, learning_rate=0.15, Nsamps = 100):
        """
        Build and optimise a Gaussian process model for the trajectory.        
        learning_rate = optimiser's learning rate
        
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        
        X = tf.Variable(np.c_[np.tile(self.obstimes,self.dims)[:,None],np.repeat(np.arange(self.dims),len(self.obstimes),axis=0)],dtype=tf.float32)
        y = tf.Variable(self.observations,dtype=tf.float32)

        #number of inducing points.
        m = self.Z.shape[0]
        #create variables that describe q(u), the variational distribution.
        mu = tf.Variable(tf.random.normal([m,1]))
        #scale = tf.Variable(tf.eye(m))#0.001*tf.random.normal([m, m])+0.1*tf.eye(m))
        scale = tf.Variable(np.tril(0.01*np.random.randn(m,m)+1*np.eye(m)),dtype=tf.float32)        

        #parameters for p(u), the prior.
        mu_u = tf.zeros([1,m])
        cov_u = tf.Variable(self.kernel.K(self.Z,self.Z))
        
        pu = tfd.MultivariateNormalFullCovariance(mu_u,cov_u+np.eye(cov_u.shape[0])*self.jitter)

        #We don't optimise the hyperparameters, so precompute.
        Kzz,Kxx,Kxz,Kzx,KzzinvKzx,KxzKzzinv,KxzKzzinvKzx = self.compute_matrices(X,self.Z)
        size = int(X.shape[0]/self.dims) #number of input points


        scalejitter = tf.eye(m)*1e-5
        for it in range(iterations):
            with tf.GradientTape() as tape:

                #the variational approximating distribution.
                qu = tfd.MultivariateNormalTriL(mu[:,0],scale+scalejitter)
                if np.any(np.isnan(qu.mean())):
                    return False #failed
                #compute the approximation over our training point locations
                #TODO only need some diagonals and off diagonal parts of qf_cov, so prob could be quicker!
                qf_mu = (KxzKzzinv @ mu)[:,0]
                qf_cov = Kxx - KxzKzzinvKzx + KxzKzzinv @ self.getcov(scale+scalejitter) @ KzzinvKzx

                #this gets us the covariance and mean for the relevant parts of the predictions. Specifically
                #a self.dims x self.dims covariance and a self.dims-element mean.
                C = tf.transpose(tf.concat([qf_cov[i::size,i::size][:,:,None] for i in range(size)],axis=2),[2,0,1])
                M = tf.transpose(tf.reshape(qf_mu,[self.dims,size]),[1,0])

                samps = tfd.MultivariateNormalTriL(M,tf.linalg.cholesky(C+tf.eye(self.dims)*self.jitter)).sample(Nsamps)
                        
                #we compute the distance from each of the observed vectors to the samples and compute their
                #log likelihoods assuming a normal distributed likelihood model over the distance from the
                #vector.
                
                #TODO This only works for 3d.
                d = tf.norm(cross(y[:,3:],samps-y[:,:3]),axis=2)/tf.norm(y[:,3:],axis=1)
                logprobs = tfd.Normal(0,self.likenoisescale).log_prob(d)
                ##ell = tf.reduce_mean(tf.reduce_sum(logprobs,1))
                ell = tfp.stats.percentile(tf.reduce_sum(logprobs,1),50,interpolation='midpoint')

                #we compute the ELBO = - (expected log likelihood of the data - KL[prior, variational_distribution]).
                elbo_loss = -( ell - tfd.kl_divergence(qu,pu) )
            #compute gradients and optimise...
            gradients = tape.gradient(elbo_loss, [mu, scale])
            optimizer.apply_gradients(zip(gradients, [mu, scale]))
            if it%20 == 0: print(it,elbo_loss.numpy())  
        self.mu = mu
        self.scale = scale
        return True

    def run(self, iterations=500, learning_rate=0.15, Nsamps = 100, jitter=1e-6):     
        self.jitter = jitter
        for jitterstep in range(5):
            if self.tryrun(iterations, learning_rate, Nsamps):
                return
            else:
                self.jitter*=10
                print("Had likely non-positive definite covariance, increasing jitter to %0.5f" % self.jitter)            
        print("Failed even with jitter of %0.5f added." % self.jitter)
            
            
