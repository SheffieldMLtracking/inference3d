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
    def __init__(self,obstimes, observations, kernel, likenoisescale, split_startend=None, Nind=None, Z=None, IndStartEnd=None, memlimit=60e6, mu=None, scale=None):
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
        Nind : number of inducing points (default is now  1+int(2*time_extent/ls))). Previously 1+int(2*np.max(obstimes))).
        memlimit : maximum number of values in a numpy array, default = 60 million (which is about 480 to 960Mb of memory).
        """
        #print("NEW PATHINFERENCE OBJECT CREATED")
        self.split_startend = split_startend
        
        if split_startend is not None:
            #we actually use the IndStartEnd as the decider for what goes in this set of observations.
            keep = (obstimes>IndStartEnd[0]) & (obstimes<=IndStartEnd[1])
            obstimes = obstimes[keep]
            observations = observations[keep]
            
        
        self.obstimes = obstimes
        self.observations = observations
        self.likenoisescale = likenoisescale
        self.dims = int(observations.shape[1]/2)
        self.memlimit = memlimit
        self.kernel = kernel
        self.mu = mu
        self.scale = scale
        
        if IndStartEnd is None:
            min_time = np.min(self.obstimes)-kernel.ls*2
            max_time = np.max(self.obstimes)+kernel.ls*2
        else:
            min_time = IndStartEnd[0]
            max_time = IndStartEnd[1]
        self.indstartend = (min_time,max_time)    
        
        if Nind is None:
            Nind = 3+int(2*(max_time-min_time)/kernel.ls)
            #print("Using %d inducing points." % Nind)
        self.Nind = Nind
        
        #Build the inducing point locations:        
        if Z is None:
            #print(self.Nind,min_time,max_time)
            #print("Spacing %d inducing points between %0.4f and %0.4f" % (self.Nind,min_time,max_time))
            self.Z = self.buildinputmatrix(self.Nind,min_time,max_time)
        else:
            print("Using prespecified inducing points")
            self.Z = Z
        #print("Z.shape")
        #print(self.Z.shape)
        #print("Using %d observations" % len(obstimes))

    def compute_matrices(self,X,Z):
        """TODO: Rename as A and B not A and Z"""
        if X.shape[0]*Z.shape[0]>self.memlimit:
            raise Exception("May exceed memory available (X x Z is %d x %d)" % (X.shape[0],Z.shape[0]))
        
        Kzz = self.kernel.K(Z,Z)+np.eye(Z.shape[0],dtype=np.float32)*1e-4 #self.jitter
        Kxx = self.kernel.K(X,X)#+np.eye(X.shape[0],dtype=np.float32)*self.jitter
        Kxz = self.kernel.K(X,Z)
        Kzx = tf.transpose(Kxz)
        KzzinvKzx = tf.linalg.solve(Kzz,Kzx)
        KxzKzzinv = tf.transpose(KzzinvKzx)
        KxzKzzinvKzx = Kxz @ KzzinvKzx
        return Kzz,Kxx,Kxz,Kzx,KzzinvKzx,KxzKzzinv,KxzKzzinvKzx
        
    
    def buildinputmatrix(self,times,min_time=None,max_time=None):
        """
        times can either be:
         - the number of times (pass an int)
         - the interval between times (pass a float)
         - or a list of times (pass a list)
        Constructs a matrix of [dims*size,2], where the first column is time and second is axis index.
        - min_time and max_time specify the linspace range of times (default to a time just before and
          after the obstimes).
        - size = number of points        
        Returns a [self.dims*size,2] matrix
        """

        #buffer_time = 0.1*(np.max(self.obstimes)-np.min(self.obstimes)) #10% of total time on either side
        buffer_time = self.kernel.ls*2 #2x lengthscale
        if max_time is None:
            max_time = np.max(self.obstimes)+buffer_time
        if min_time is None:
            min_time = np.min(self.obstimes)-buffer_time
        A = []
           
        if isinstance(times,int):
            #print("Building matrix with %d items" % times)
            size = times
            for ax in range(self.dims):
                Aax = np.c_[np.linspace(min_time,max_time,size),np.full(size,ax)]
                A.extend(Aax)
            A = np.array(A)
            A = tf.Variable(A,dtype=tf.float32)    
            return A
        elif isinstance(times,float):    
            #print("Building matrix with spacing of %0.3f seconds." % times)
            for ax in range(self.dims):
                ts = np.arange(min_time,max_time,times)
                Aax = np.c_[ts,np.full(len(ts),ax)]
                A.extend(Aax)
            A = np.array(A)
            A = tf.Variable(A,dtype=tf.float32)    
            return A
        else:
            times = np.array(times)
            #print("Building matrix with prespecified times (length = %d)" % len(times))
            assert len(times.shape)==1, "Expecting a 1d list of times, or the number of time points."
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
    
    def tryrun(self, iterations=500, learning_rate=0.15, Nsamps = 100,mu=None,scale=None):
        """
        Build and optimise a Gaussian process model for the trajectory.        
        learning_rate = optimiser's learning rate
        
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        if mu is None: mu = self.mu
        if scale is None: scale = self.scale
        
        
        
        X = tf.Variable(np.c_[np.tile(self.obstimes,self.dims)[:,None],np.repeat(np.arange(self.dims),len(self.obstimes),axis=0)],dtype=tf.float32)
        y = tf.Variable(self.observations,dtype=tf.float32)

        #number of inducing points.
        m = self.Z.shape[0]
        #create variables that describe q(u), the variational distribution.
        if mu is None:
            #print("Mu is not set... initialising random normal")
            mu = tf.Variable(tf.random.normal([m,1])*0)
        else:
            mu = tf.Variable(mu)            
        if scale is None:
            scale = tf.Variable(tf.eye(m)*0.01) #0.001*tf.random.normal([m, m])+0.1*tf.eye(m))
        else:
            scale = tf.Variable(scale)
            

        #scale = tf.Variable(np.tril(0.01*np.random.randn(m,m)+1*np.eye(m)),dtype=tf.float32)        

        #parameters for p(u), the prior.
        mu_u = tf.zeros([1,m])
        #print(mu_u)

        for prior_jitter in [1e-3]: # [1e-6,1e-5,1e-4,1e-3]:
            cov_u = tf.Variable(self.kernel.K(self.Z,self.Z)+np.eye(m)*prior_jitter)
            if (np.linalg.det(cov_u)>=0) and (np.linalg.det(cov_u)<np.inf): break
          
        print("Added jitter to prior covariance of %0.8f" % prior_jitter)

        #print("DETERMINATE OF PRIOR COVARIANCE!")
        #print(np.linalg.det(cov_u))
        #print("EIGEN VALUES")
        #print(np.linalg.eig(cov_u))
        #print(cov_u)
        
        pu = tfd.MultivariateNormalFullCovariance(mu_u,cov_u)#+np.eye(cov_u.shape[0])*self.jitter)

        #We don't optimise the hyperparameters, so precompute.
        Kzz,Kxx,Kxz,Kzx,KzzinvKzx,KxzKzzinv,KxzKzzinvKzx = self.compute_matrices(X,self.Z)
        size = int(X.shape[0]/self.dims) #number of input points

        elbo_record = []

        #scalejitter = tf.eye(m)*1e-5
        for it in range(iterations):
            with tf.GradientTape() as tape:

                #the variational approximating distribution.
                #jit = tf.eye(m)*self.jitter
                qu = tfd.MultivariateNormalTriL(mu[:,0],scale)#+scalejitter)
                if np.any(np.isnan(qu.mean())):
                    #print("Iteration %4d. ELL=%9.1f Prior=%9.1f. Total ELBO Loss=%9.1f" % (it,ell.numpy(), tfd.kl_divergence(qu,pu).numpy(), elbo_loss.numpy()))
                    return False #failed
                #compute the approximation over our training point locations
                #TODO only need some diagonals and off diagonal parts of qf_cov, so prob could be quicker!
                qf_mu = (KxzKzzinv @ mu)[:,0]
                #qf_cov = Kxx - KxzKzzinvKzx + KxzKzzinv @ self.getcov(scale+self.jitter) @ KzzinvKzx #scalejitter
                qf_cov = Kxx - KxzKzzinvKzx + KxzKzzinv @ self.getcov(scale) @ KzzinvKzx #scalejitter                

                #this gets us the covariance and mean for the relevant parts of the predictions. Specifically
                #a self.dims x self.dims covariance and a self.dims-element mean.
                C = tf.transpose(tf.concat([qf_cov[i::size,i::size][:,:,None] for i in range(size)],axis=2),[2,0,1])
                M = tf.transpose(tf.reshape(qf_mu,[self.dims,size]),[1,0])

                #too much jitter!
                #print(M,C)

                #print("-----------------------")
                #for i in range(C.shape[0]):
                #    print(i)
                #    print(C[i,:,:])
                #    print("cholesky...")
                #    print(tf.linalg.cholesky(C[i,:,:]+tf.eye(self.dims)*self.jitter))
                #    print(tfd.MultivariateNormalFullCovariance(M[i,:],C[i,:,:]+tf.eye(self.dims)*self.jitter).sample(1))
                #print("Trying cholesky on whole thing...")
                #print(tf.linalg.cholesky(C+tf.eye(self.dims)*self.jitter))
                #print("Sampling...")
                #print(np.linalg.det((C+tf.transpose(C,perm=[0,2,1]))/2+tf.eye(self.dims)*self.jitter))
                
                C = (C+tf.transpose(C,perm=[0,2,1]))/2 +tf.eye(self.dims)*self.jitter
                #print(np.linalg.det(C))
                
                #samps = tfd.MultivariateNormalFullCovariance(M,C).sample(Nsamps)               
                #print("Sampling...")                
                samps = tfd.MultivariateNormalTriL(M,tf.linalg.cholesky(C)).sample(Nsamps)
                #print("Done")
                #we compute the distance from each of the observed vectors to the samples and compute their
                #log likelihoods assuming a normal distributed likelihood model over the distance from the
                #vector.
                
                #TODO This only works for 3d.
                d = tf.norm(cross(y[:,3:],samps-y[:,:3]),axis=2)/tf.norm(y[:,3:],axis=1)
                logprobs = tfd.Normal(0,self.likenoisescale).log_prob(d)                
                ell = tf.reduce_mean(tf.reduce_sum(logprobs,1))
                
                #print(ell,tfd.kl_divergence(qu,pu))
                ##ell = tfp.stats.percentile(tf.reduce_sum(logprobs,1),50,interpolation='midpoint')

                #we compute the ELBO = - (expected log likelihood of the data - KL[prior, variational_distribution]).
                
                elbo_loss = -( ell - tfd.kl_divergence(qu,pu) )
            #compute gradients and optimise...
            if it>0: #we're going to skip optimising the first iteration (so we can return logprobs etc if we want, by setting Nsteps=1).
                gradients = tape.gradient(elbo_loss, [mu, scale])
                optimizer.apply_gradients(zip(gradients, [mu, scale]))
            if it%20 == 0: 
                print("Iteration %4d. ELL=%9.1f Prior=%9.1f. Total ELBO Loss=%9.1f" % (it,ell.numpy(), tfd.kl_divergence(qu,pu).numpy(), elbo_loss.numpy()))
            elbo_record.append(elbo_loss.numpy())
            #if it>500:
            #    delta = np.mean(elbo_record[-40:]) - np.mean(elbo_record[-80:-40]) #-20
            #    #print(delta,-np.std(elbo_record[-64:])/8)
            #    #if change in rolling mean < one standard error, then we quit.
            #    if delta>-np.std(elbo_record[-64:])/8: #standard error
            #        print("Probably optimised. Stopping early.")
            #        print("Last 40 ELBO values:")
            #        print(", ".join(["%0.0f" % v for v in elbo_record[-40:]]))
            #        break
        self.mu = mu
        self.last_logprobs = logprobs
        self.scale = scale
        return True

    def run(self, iterations=500, learning_rate=0.15, Nsamps = 100, jitter=0.001,mu=None,scale=None):     
        self.jitter = jitter
        for jitterstep in range(5):
            if self.tryrun(iterations, learning_rate, Nsamps,mu=mu,scale=scale):
                return
            else:
                self.jitter*=10
                print("Had likely non-positive definite covariance, increasing jitter to %0.5f" % self.jitter)    
                #assert False        
        print("Failed even with jitter of %0.5f added." % self.jitter)
        raise Exception("Failed even with jitter of %0.5f added." % self.jitter)
            
