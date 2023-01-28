# Kratoscope

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

## Création d'un échantillon ###################################################################

def create_sample(shape, type):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    u,v,w = shape
    x = np.linspace(-u,u,1000)
    y = np.linspace(-v,v,1000)
    xx, yy = np.meshgrid(x,y)
    if type == 'curve':
        # elliptic paraboloid
        zz = xx**2 + yy**2
        surf = ax.plot_surface(xx, yy, zz, cmap='turbo', linewidth=0, antialiased=False)
    elif type == 'plane':
        # two planes distant of 5 
        zz = np.zeros((1000,1000,2))
        zz[:,:,0] = xx
        zz[:,:,1] = xx + 5
        surf = ax.plot_surface(xx, yy, zz[:,:,0], cmap='turbo', linewidth=0, antialiased=False)
        surf = ax.plot_surface(xx, yy, zz[:,:,1], cmap='turbo', linewidth=0, antialiased=False)
    else :
        raise Exception('Type not valid')
    
    plt.show()
    return xx, yy, zz
#X, Y, Z = create_sample((10,10,10), 'curve')

def create_sample_matrix(N, P, Q, type):
    sample = np.zeros((N,P,Q))
    for q in range(Q):
        if type=='paraboloid':
            # Paraboloid : R = sqrt(q), R < sqrt(x^2 + y^2) < R+1
            D = np.array([(q <= (n-N/2)**2 + (p-P/2)**2 < 2*np.sqrt(q)+q+1) for n in range(N) for p in range(P)])
            D = np.where(D, 1, 0)
            sample[:,:,q] = D.reshape((N,P))
        elif type=='hyperboloid':
            # Hyperboloid : R = q, R < sqrt(x^2 + y^2) < R+1
            D = np.array([(q <= np.sqrt((n-N/2)**2 + (p-P/2)**2) < q+1) for n in range(N) for p in range(P)])
            D = np.where(D, 1, 0) # True -> 1
            sample[:,:,q] = D.reshape((N,P))
        elif type=='plane':
            sample[:,:,q] = np.eye(N,P,q) + np.eye(N,P,q+3)
    return sample

def affiche_1(matrix):
    for q in range(matrix.shape[2]):
        plt.figure()
        plt.imshow(matrix[:,:,q], cmap='gray', vmin=0, vmax=1)
        plt.title('n° {}/{}'.format(matrix.shape[2]-q, matrix.shape[2]))
        plt.xlim(-1,matrix.shape[0])
        plt.ylim(-1,matrix.shape[1])
        plt.show()

def affiche_3(sample, ker, res):
    ''' Numérotation des coupes pas forcèment cohérente'''
    for q in range(sample.shape[2]):
        fig, ax = plt.subplots(1,3, sharey=True, sharex=True)
        ax[0].imshow(sample[:,:,q], cmap='gray', vmin=0, vmax=1)
        ax[0].set_title('n° {}/{}'.format(sample.shape[2]-q, sample.shape[2]))
        ax[0].set_xlim(-1,ker.shape[0])
        ax[0].set_ylim(-1,ker.shape[1])

        # Garde-fou : ker.shape != sample.shape
        ax[1].imshow(ker[:,:,min(q,ker.shape[2])], cmap='gray', vmin=0, vmax=1)
        ax[1].set_title('n° {}/{}'.format(ker.shape[2]-q, ker.shape[2]))

        # Garde-fou : res.shape != sample.shape
        ax[2].imshow(res[:,:,min(q,res.shape[2])], cmap='gray', vmin=0, vmax=1)
        ax[2].set_title('n° {}/{}'.format(res.shape[2]-q, res.shape[2]))

        plt.show()

## Simulation de l'observation (Convolution 3D) ################################################

def kernel_old(k, Q, sigma, sigma_0, amp, type):
    # Use sig.convolve(sample, ker, 'same')
    N, P = 2*k+1, 2*k+1
    ker = np.zeros((N,P,Q))
    #ker[k,k,int(Q/2)] = 1
    for q in range(int(Q/2),Q):
        if type=='scatter':
            # energie ? /(2*np.pi*q*sigma**2)
            ker[:,:,q] = np.array([np.exp(-((n-k)**2 + (p-k)**2)/(2*(sigma_0**2 + (q-int(Q/2))*(sigma**2)))) for n in range (N) for p in range(P)]).reshape((N,P))
        elif type=='attenuation':
            ker[k,k,q] = (amp**(q-int(Q/2)))/(2*np.pi*(sigma_0**2 + (q-int(Q/2))*(sigma**2)))
            #ker[:,:,q] = np.ones((N,P))*(amp**(q-int(Q/2)))/(2*np.pi*(sigma_0**2 + (q-int(Q/2))*(sigma**2)))
        elif type=='both':
            ker[:,:,q] = np.array([np.exp(-((n-k)**2 + (p-k)**2)/(2*(sigma_0**2 + (q-int(Q/2))*(sigma**2))))
            *(amp**(q-int(Q/2)))/(2*np.pi*(sigma_0**2 + (q-int(Q/2))*(sigma**2))) for n in range (N) for p in range(P)]).reshape((N,P))
        else :
            raise Exception('Type not valid')
    return ker

def kernel(k, Q, sigma, sigma_0, amp, type):
    ''' Use sig.convolve(sample, ker, 'full') 
        <=> épaisseur de Q_k-1 de paraffin sans échantillon sur les bords'''
    N, P = 2*k+1, 2*k+1
    ker = np.zeros((N,P,Q))
    #ker[k,k,int(Q/2)] = 1
    for q in range(Q):
        if type=='scatter':
            # energie ? /(2*np.pi*q*sigma**2)
            ker[:,:,q] = np.array([np.exp(-((n-k)**2 + (p-k)**2)/(2*(sigma_0**2 + q*(sigma**2)))) for n in range (N) for p in range(P)]).reshape((N,P))
        elif type=='attenuation':
            ker[k,k,q] = amp**q/(2*np.pi*(sigma_0**2 + (q-int(Q/2))*(sigma**2)))
            #ker[:,:,q] = np.ones((N,P))*(amp**(q-int(Q/2)))/(2*np.pi*(sigma_0**2 + (q-int(Q/2))*(sigma**2)))
        elif type=='both':
            ker[:,:,q] = np.array([np.exp(-((n-k)**2 + (p-k)**2)/(2*(sigma_0**2 + q*(sigma**2))))
            *(amp**q/(2*np.pi*(sigma_0**2 + q*(sigma**2)))) for n in range (N) for p in range(P)]).reshape((N,P))
        else :
            raise Exception('Type not valid')
    return ker

def noise(sig, snr):
    E_sig = np.sum(sig**2)
    sigma_noise = np.sqrt((E_sig*(10**(-snr/10)))/np.prod(sig.shape))
    sig_noise = sig + sigma_noise*np.random.rand(sig.shape[0],sig.shape[1],sig.shape[2])
    E_noise = np.sum((sig-sig_noise)**2)
    print('E_noise = ', E_noise)
    print('E_sig = ', E_sig)
    #print(sigma_noise)
    print('N*sigma = ',(sigma_noise**2)*np.prod(sig.shape))
    print('SNR = ', 10*np.log10(E_sig/E_noise))
    return sig_noise

N, P, Q = 50, 50, 25
sample = create_sample_matrix(N, P, Q, 'hyperboloid')
#affiche_1(sample)

k = 5
Q_k = 5 # Deepness of the PSF
sigma = 1
sigma_0 = 0.5
amp = 0.9 # <1
#ker1 = kernel_old(k, 2*Q_k+1, sigma, sigma_0, amp, 'both')
ker1 = kernel(k, Q_k, sigma, sigma_0, amp, 'both')
print(ker1.shape)
#affiche_1(ker1)

def Experiment(sample, ker, maxIter, conv_mode, noisy, snr, verbose):
    ''' Full hypothesis: sampling complete, parafin at begining and end
        Same hypothesis: sampling not complete, no parafin at the end '''
    
    sim = sig.convolve(sample[:,:,::-1], ker, conv_mode) 
    if noisy:
        sim = noise(sim,snr)
    res = ISRA(sim, ker, maxIter, conv_mode)
    if verbose:
        affiche_1(sim[:,:,::-1])
        affiche_3(sample, sim[:,:,::-1], res[:,:,::-1])

    return sim, res

# Affichage sympa pour les rapports
def nice_subplots(N,P,Q,mode):
    ''' Ne fonctionne pas en mode 'sample' si Q =< 20, pour 'simul' il y a aussi une limite'''
    fig, ax = plt.subplots(3,5,sharey=True, sharex=True)
    fig.set_figheight(9)
    fig.set_figwidth(15)
    i = 0
    if mode == 'sample':
        for type in ['paraboloid', 'hyperboloid', 'plane']:
            sample = create_sample_matrix(N, P, Q, type)
            for j in range(5):
                ax[i,j].imshow(sample[:,:,5*j], cmap='gray', vmin=0, vmax=1)
                ax[i,j].axis('off')
                if not j:
                    ax[i,j].set_title('{} : n° {}/{}'.format(type, sample.shape[2]-5*j, sample.shape[2]))
                else : 
                    ax[i,j].set_title('n° {}/{}'.format(sample.shape[2]-5*j, sample.shape[2]))
            i += 1
    elif mode =='kernel':
        k = 5
        Q_k = 10
        sigma = 0.8
        sigma_0 = 0.9
        amp = 1
        for i in range(3):
            ker = 9*kernel(k, (i*Q_k+25), (i+1)*sigma, sigma_0, amp, 'both')
            for j in range(5):
                ax[i,j].imshow(ker[:,:,3*j], cmap='gray', vmin=0, vmax=1)
                ax[i,j].axis('off')
                ax[i,j].set_title('n° {}/{}'.format(ker.shape[2]-3*j, ker.shape[2]))
    elif mode == 'simul':
        k = 5
        Q_k = 10
        sigma = 0.8
        sigma_0 = 1.1
        amp = 0.8
        i = 0
        for type in ['paraboloid', 'hyperboloid', 'plane']:
            sample = create_sample_matrix(N, P, Q, type)
            ker = kernel(k, (i*Q_k+25), (3-i)*sigma, sigma_0, amp, 'both')
            sim = sig.convolve(sample, ker, 'full')
            for j in range(5):
                ax[i,j].imshow(sim[:,:,5*j], cmap='gray', vmin=0, vmax=1)
                ax[i,j].axis('off')
                if not j:
                    ax[i,j].set_title('{} : n° {}/{}'.format(type, sim.shape[2] - 5*j, sim.shape[2]))
                else : 
                    ax[i,j].set_title('n° {}/{}'.format(sim.shape[2] - 5*j, sim.shape[2]))
            i += 1
    else :
        raise Exception('Mode not valid')

    plt.show()

#nice_subplots(N,P,20,'simul')

## Algorithme ISRA ###########################################################################
def ISRA(y, H, maxIter, conv_mode):
    ''' Input : y is a 3D image, H is a 3D kernel and maxIter a scalar
        Output : x is a 3D image '''

    # Initialisation de x (ne pas initialiser à 0)
    if conv_mode == 'same':
        x = np.ones((y.shape))
    elif conv_mode == 'full':
        N, P, Q = y.shape
        k1, k2, Q_k = H.shape
        x = np.ones((N-k1+1, P-k2+1, Q-Q_k+1)) 
    
    compt_iter = 0
    err_list = np.zeros((maxIter,))
    while compt_iter < maxIter: # Stop condition ?
        if conv_mode == 'same':
            err_list[compt_iter] = error(sig.convolve(x, H, "same"), y)
        elif conv_mode == 'full':
            err_list[compt_iter] = error(sig.convolve(x, H, "full"), y)
        print('Error = ', err_list[compt_iter])
        if not(np.all(x>0)):
            raise Exception('Non positive element detected')
        u = U(y, H, conv_mode)
        v = V(x, y, H, conv_mode)
        # Maximal Step Size
        alpha = init_stepsize(u, v, x, y, H, conv_mode)
        print("Initial stepsize = {}".format(alpha))
        # Descent Direction 
        d = descent_direction(x, u, v)
        # Optimal Step
        #alpha = armijo(alpha_0, beta, gamma)
        # Next estimate
        x = x + alpha*d
        compt_iter += 1

    plt.figure()
    plt.plot(np.linspace(1,maxIter-1,maxIter-1), err_list[1::])
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()
    return x

def grad_J(x, y, H, conv_mode):
    ''' x and y are 3D images and H is a 3D kernel
        output : 3D array of x.size'''
    H_flip = np.flip(H)
    if conv_mode == 'full':
        Hx = sig.convolve(x, H, 'full')
        HtHx = sig.convolve(Hx, H_flip, 'valid')
        Hty = sig.convolve(y, H_flip, 'valid')
    elif conv_mode == 'same':
        Hx = sig.convolve(x, H, 'same')
        HtHx = sig.convolve(Hx, H_flip, 'same')
        Hty = sig.convolve(y, H_flip, 'same')
    return HtHx - Hty

def U(y, H, conv_mode):
    ''' y is a 3D image and H is a 3D kernel
        output : 3D array of x.size'''
    H_flip = np.flip(H)
    if conv_mode == 'full':
        Hty = sig.convolve(y, H_flip, 'valid')
    elif conv_mode == 'same':
        Hty = sig.convolve(y, H_flip, 'same')
    return np.where(Hty>0, Hty, 0)

def V(x, y, H, conv_mode):
    ''' x and y are 3D images and H is a 3D kernel
        output : 3D array of x.size'''
    H_flip = np.flip(H)
    if conv_mode == 'full':
        Hx = sig.convolve(x, H, 'full')
        HtHx = sig.convolve(Hx, H_flip, 'valid')
        Hty = sig.convolve(y, H_flip, 'valid')
    elif conv_mode == 'same':
        Hx = sig.convolve(x, H, 'same')
        HtHx = sig.convolve(Hx, H_flip, 'same')
        Hty = sig.convolve(y, H_flip, 'same')
    return HtHx - np.where(Hty<0, Hty, 0)

def init_stepsize(u, v, x, y, H, conv_mode):
    ''' Input : 3D arrays
        Output : scalar '''
    uv = 1/(1-u/(v+1e-8))
    grad = grad_J(x, y, H, conv_mode)
    try :
        return np.min(uv[(grad>0) & (x>0)])
    except Exception:
        print("No satisfying value encountered")
        return 1

def descent_direction(x, u, v):
    ''' Input : 3D arrays
        Output : 3D array '''
    return (x/(v+1e-8))*(u-v)

def error(res, sample):
    # Norme euclidienne
    res = np.ravel(res)
    sample = np.ravel(sample)
    return (np.sum((res-sample)**2))/np.sum(sample**2)

sim, res = Experiment(sample, ker1, maxIter=50, conv_mode='same', noisy=True, snr=30, verbose=False)
#res_error = error(res, sample)
#print("error = {:5f}".format(res_error))

#affiche_3(sample, sim1_noise, res) #Doesn't work with 'same'
#affiche_3(sample, sim1[:,:,::-1], res[:,:,::-1])