from scipy.stats import Normal
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


class ParticleFilter:
    def __init__(self,lims,obstimes,observations,Nparticles=100000,std_onesec=1.0,init_speed_std=2):
        """
        lims = domain [[x_start, x_end],[y_start, y_end],[z_start,z_end]]
        obstimes = observation times
        observations = ..        
        """

        indices = np.argsort(obstimes)
        observations = observations[indices,:]
        obstimes = obstimes[indices]

        self.ndims = len(lims)
        self.lims = np.array(lims)
        self.obstimes = obstimes
        self.observations = observations
        self.Nparticles = Nparticles
        self.particles = np.random.rand(Nparticles, self.ndims*2)
        self.particles[:,:self.ndims]=self.particles[:,:self.ndims]*(self.lims[:,1]-self.lims[:,0])+self.lims[:,0]
        self.particles[:,self.ndims:(self.ndims*2)] = np.random.randn(Nparticles,self.ndims)*init_speed_std #random velocity...
        self.std_onesec = std_onesec #metres/second std after 1 second?

    
    def run(self,store_given_yn=True):
        means = []
        covs = []
        lastobstime = self.obstimes[0]
        times = []
        for i,(obstime, obs) in enumerate(zip(self.obstimes,self.observations)):
            if i%10==0:
                print("%4d/%4d" % (i,len(self.obstimes)),end="\r")
            deltatime = obstime - lastobstime
            lastobstime = obstime

            if not store_given_yn:
                if np.isnan(obs[0]):
                    means.append(np.mean(self.particles,0))
                    covs.append(np.cov(self.particles.T)) #the mean and cov of p(x_n|y_1:n-1)
                
            if not np.isnan(obs[0]):
                ps = self.get_probabilities(obstime,obs)
                
                keep = np.random.choice(self.Nparticles,size=self.Nparticles,p=ps)
                self.particles = self.particles[keep,:] #p(x_n|y_1:n)
                
            if store_given_yn:
                if np.isnan(obs[0]):
                    means.append(np.mean(self.particles,0))
                    covs.append(np.cov(self.particles.T)) #the mean and cov of p(x_n|y_1:n)
                    times.append(obstime)
            
            
            self.particles[:,:self.ndims] += self.particles[:,self.ndims:self.ndims*2]*deltatime
            self.particles[:,self.ndims:self.ndims*2]+=np.random.randn(self.Nparticles,self.ndims)*np.sqrt(deltatime)*self.std_onesec
        self.means = np.array(means)
        self.covs = np.array(covs)
        self.times = np.array(times)
        print("")
        
    def get_probabilities(self,obstime,obs):
        raise NotImplementedError

class ParticleFilterSingleVector(ParticleFilter):
    
    def __init__(self,lims,obstimes,observations,Nparticles=100000,std_onesec=1.0,init_speed_std=2,likelihood_std=0.15):
        super().__init__(lims,obstimes,observations,Nparticles=100000,std_onesec=std_onesec,init_speed_std=init_speed_std)
        self.norm_pdf = Normal(mu = 0, sigma = likelihood_std)
        
    def get_probabilities(self,obstime,obs):
        d = np.linalg.norm(np.cross(self.particles[:,0:self.ndims]-obs[:self.ndims],obs[self.ndims:]),axis=1)
        ps = self.norm_pdf.pdf(d)
        return ps/np.sum(ps)

def cov_prod(mu1,mu2,cov1,cov2):
    cov1add2inv = np.linalg.inv(cov1+cov2)
    cov = cov1@cov1add2inv@cov2
    mu = cov2@cov1add2inv@mu1 + cov1@cov1add2inv@mu2
    return mu,cov

class Smooth:
    def __init__(self,obstimes,observations,lims,Nparticles=10000,times=None,means=None,covs=None,init_speed_std=2,likelihood_std=0.15,std_onesec=1.0):
        
        
        #need to add test times!
        test_times = np.arange(np.floor(np.min(obstimes)),np.ceil(np.max(obstimes)),0.1)
        obstimes = np.r_[obstimes,test_times]
        observations = np.r_[observations,np.full((len(test_times),6),np.nan)]
        indices = np.argsort(obstimes)
        observations = observations[indices,:]
        obstimes = obstimes[indices]
        
        self.obstimes = obstimes
        self.observations = observations
        
        self.lims = lims
        self.Nparticles = Nparticles
        self.ndims = len(lims)
        self.means = means
        self.times = times
        self.covs = covs
        self.likelihood_std = likelihood_std
        self.init_speed_std=init_speed_std
        self.std_onesec=std_onesec
       
        
    def run(self):
        pf_forwards = ParticleFilterSingleVector(self.lims,self.obstimes,self.observations,Nparticles=self.Nparticles,likelihood_std=self.likelihood_std,init_speed_std=self.init_speed_std,std_onesec=self.std_onesec)
        pf_forwards.run()


        pf_backwards = ParticleFilterSingleVector(self.lims,-self.obstimes,self.observations,Nparticles=self.Nparticles,likelihood_std=self.likelihood_std,init_speed_std=self.init_speed_std,std_onesec=self.std_onesec)
        pf_backwards.run(store_given_yn=False)

        newmeans = []
        #newstds = []
        newcovs = []
        newtimes = []
        for i,(time,forward_mean,forward_cov,backward_mean,backward_cov) in enumerate(zip(pf_forwards.times,pf_forwards.means,pf_forwards.covs,pf_backwards.means[::-1,:],pf_backwards.covs[::-1,:,:])):
            mu,cov = cov_prod(forward_mean, backward_mean, forward_cov, backward_cov)
            newmeans.append(mu)
            #newstds.append(np.sqrt(np.sum(np.diag(cov[:2,:2]))))
            newcovs.append(cov)
            newtimes.append(time)
        self.means = np.array(newmeans)
        self.covs = np.array(newcovs)
        self.times = np.array(newtimes)


    def getpredictions(self,pred_times):
        """
        We'll linearly interpolate the means, and simply report the nearest covariance (interpolating covariances is harder).
        """
        predmeans = np.array([np.interp(pred_times,self.times,self.means[:,dim]) for dim in range(self.ndims*2)]).T
        predcovs = self.covs[np.argmin(np.abs(self.times[:,None]-pred_times[None,:]),axis=0),:,:]
        print("!!")
        print(predmeans.shape, predcovs.shape)
        return predmeans, predcovs
        
def confidence_ellipse(mean,cov, ax, n_std=1.0, facecolor='none', edgecolor='black', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    #print(ell_radius_x,ell_radius_y)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
