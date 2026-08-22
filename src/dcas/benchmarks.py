import numpy as np

class Problem:
    def __init__(self, name, dim, bounds, f, g, scales=None):
        self.name=name; self.dim=dim; self.bounds=np.asarray(bounds,float); self.f=f; self.g=g
        self.scales=np.ones(1) if scales is None else np.asarray(scales,float)
    def evaluate(self,x):
        x=np.asarray(x,float); return float(self.f(x)), np.asarray(self.g(x),float)
    def cv(self,g): return float(np.maximum(g,0).sum())
    def ncv(self,g):
        s=self.scales if len(self.scales)==len(g) else np.ones(len(g))
        return float((np.maximum(g,0)/np.maximum(s,1e-12)).sum())
    def feasible(self,g,tol=1e-10): return bool(np.all(np.asarray(g)<=tol))

def g06():
    f=lambda x:(x[0]-10)**3+(x[1]-20)**3
    g=lambda x:np.array([-(x[0]-5)**2-(x[1]-5)**2+100,(x[0]-6)**2+(x[1]-5)**2-82.81])
    return Problem('G06',2,[[13,100],[0,100]],f,g,[100,82.81])

def g08():
    f=lambda x:-(np.sin(2*np.pi*x[0])**3*np.sin(2*np.pi*x[1]))/(x[0]**3*(x[0]+x[1])+1e-12)
    g=lambda x:np.array([x[0]**2-x[1]+1,1-x[0]+(x[1]-4)**2])
    return Problem('G08',2,[[0.1,10],[0.1,10]],f,g,[100,100])

def g11():
    f=lambda x:x[0]**2+(x[1]-1)**2
    g=lambda x:np.array([abs(x[1]-x[0]**2)-1e-3])
    return Problem('G11',2,[[-1,1],[-1,1]],f,g,[1])

def annulus10():
    d=10
    f=lambda x:np.sum((x-0.35)**2)+0.05*np.sum(np.cos(8*x))
    def g(x):
        r=np.sum(x*x)
        return np.array([1.2-r,r-3.2,0.15-np.mean(x),0.10-np.mean(np.sin(3*x))])
    return Problem('Annulus10',d,[[-2,2]]*d,f,g,[3.2,3.2,2,2])

def rotated_box10(seed=123):
    rng=np.random.default_rng(seed); A=rng.normal(size=(10,10)); Q,_=np.linalg.qr(A); c=np.linspace(-0.25,0.25,10)
    f=lambda x:np.sum((x-0.6)**2)+2*np.sum(np.sin(2.5*x)**2)
    def g(x):
        z=Q@(x-c); return np.r_[z-0.45,-z-0.45]
    return Problem('RotatedBox10',10,[[-2,2]]*10,f,g,[2]*20)

def welded_beam():
    P=6000.; L=14.; E=30e6; G=12e6; tau_max=13600.; sigma_max=30000.; delta_max=0.25
    f=lambda x:1.10471*x[0]**2*x[1]+0.04811*x[2]*x[3]*(14+x[1])
    def g(x):
        h,l,t,b=x
        tau1=P/(np.sqrt(2)*h*l); M=P*(L+l/2); R=np.sqrt(l*l/4+(h+t)**2/4)
        J=2*np.sqrt(2)*h*l*(l*l/12+(h+t)**2/4); tau2=M*R/(J+1e-12)
        tau=np.sqrt(tau1*tau1+2*tau1*tau2*l/(2*R+1e-12)+tau2*tau2)
        sigma=6*P*L/(b*t*t+1e-12); delta=4*P*L**3/(E*t**3*b+1e-12)
        pc=4.013*E*np.sqrt(t*t*b**6/36)/(L**2)*(1-t/(2*L)*np.sqrt(E/(4*G)))
        return np.array([tau-tau_max,sigma-sigma_max,h-b,0.125-h,delta-delta_max,P-pc])
    return Problem('WeldedBeam',4,[[0.1,2],[0.1,10],[0.1,10],[0.1,2]],f,g,[tau_max,sigma_max,2,2,delta_max,P])

def narrow_corridor12():
    d=12
    f=lambda x:np.sum((x-0.25)**2)+0.1*np.sum(np.sin(6*x)**2)
    def g(x):
        s=np.sum(x); q=np.sum((x-0.1)**2)
        return np.array([abs(s-1.2)-0.18, q-2.2, 0.15-np.mean(np.cos(2*x))])
    return Problem('NarrowCorridor12',d,[[-1.5,1.5]]*d,f,g,[3,3,2])

def two_basin8():
    d=8; c1=np.full(d,-0.55); c2=np.full(d,0.6)
    f=lambda x:min(np.sum((x-c1)**2)+0.8,np.sum((x-c2)**2)) + 0.03*np.sum(np.cos(10*x))
    def g(x):
        r1=np.sum((x-c1)**2)-0.75; r2=np.sum((x-c2)**2)-0.55
        return np.array([r1*r2])
    return Problem('TwoBasin8',d,[[-1.5,1.5]]*d,f,g,[4])

def rotated_ellipsoid20(seed=77):
    d=20; rng=np.random.default_rng(seed); Q,_=np.linalg.qr(rng.normal(size=(d,d))); c=np.linspace(-0.2,0.2,d)
    weights=np.geomspace(1,20,d)
    f=lambda x:np.sum(weights*(Q@(x-0.4))**2)
    def g(x):
        z=Q@(x-c)
        return np.array([np.sum((z/0.55)**2)-d, 0.1-np.mean(z[:5])])
    return Problem('RotatedEllipsoid20',d,[[-2,2]]*d,f,g,[d,2])

def sparse_feasible20():
    d=20; target=np.linspace(-0.3,0.3,d)
    f=lambda x:np.sum((x-target)**2)+0.05*np.sum(np.sin(5*x)**2)
    def g(x):
        return np.array([abs(np.sum(x[:10]))-0.3, abs(np.sum(x[10:]))-0.3, np.sum((x-target)**2)-5.0])
    return Problem('SparseFeasible20',d,[[-2,2]]*d,f,g,[5,5,10])

def chain_constraint30():
    d=30
    f=lambda x:np.sum((x-0.15)**2)
    def g(x):
        pair=x[:-1]+0.55*x[1:]-0.35
        return np.r_[pair, np.sum(x*x)-8.0]
    return Problem('ChainConstraint30',d,[[-1.5,1.5]]*d,f,g,[2]*29+[10])

def multishell12():
    d=12; c=np.full(d,0.25)
    f=lambda x:np.sum((x-c)**2)+0.04*np.sum(np.cos(9*x))
    def g(x):
        r=np.sum((x-c)**2)
        shell=(r-1.0)*(r-2.4)
        return np.array([shell, 0.08-np.mean(x), np.mean(x)-0.75])
    return Problem('MultiShell12',d,[[-1.5,1.5]]*d,f,g,[4,2,2])

def all_problems():
    return [g06(),g08(),g11(),annulus10(),rotated_box10(),welded_beam(),narrow_corridor12(),two_basin8(),rotated_ellipsoid20(),sparse_feasible20(),chain_constraint30(),multishell12()]

def problem_by_name(name):
    for p in all_problems():
        if p.name==name: return p
    raise KeyError(name)

# Additional engineering and scalable benchmarks used in the confirmatory extension.
def pressure_vessel():
    # Continuous relaxation of the classical pressure-vessel design problem.
    def f(x):
        x1,x2,x3,x4=x
        return 0.6224*x1*x3*x4 + 1.7781*x2*x3*x3 + 3.1661*x1*x1*x4 + 19.84*x1*x1*x3
    def g(x):
        x1,x2,x3,x4=x
        return np.array([
            0.0193*x3-x1,
            0.00954*x3-x2,
            1296000.0-np.pi*x3*x3*x4-(4.0/3.0)*np.pi*x3**3,
            x4-240.0,
        ])
    return Problem('PressureVessel',4,[[0.0625,6.1875],[0.0625,6.1875],[10,200],[10,240]],f,g,[6,6,1.3e6,240])

def tension_spring():
    # Classical tension/compression spring design benchmark (continuous form).
    def f(x):
        d,D,N=x
        return (N+2.0)*D*d*d
    def g(x):
        d,D,N=x
        den=max(12566.0*(D*d**3-d**4),1e-12)
        return np.array([
            1.0-(D**3*N)/(71785.0*d**4+1e-12),
            (4.0*D**2-d*D)/den + 1.0/(5108.0*d*d+1e-12)-1.0,
            1.0-(140.45*d)/(D**2*N+1e-12),
            (d+D)/1.5-1.0,
        ])
    return Problem('TensionSpring',3,[[0.05,2.0],[0.25,1.3],[2.0,15.0]],f,g,[2,2,2,2])

def cantilever_beam():
    # Five-segment cantilever design benchmark.
    coeff=np.array([61.,37.,19.,7.,1.])
    def f(x): return 0.0624*np.sum(x)
    def g(x): return np.array([np.sum(coeff/np.maximum(x,1e-12)**3)-1.0])
    return Problem('CantileverBeam',5,[[0.01,100.0]]*5,f,g,[10])

def scalable_chain(dim=100):
    d=int(dim)
    target=np.full(d,0.15)
    def f(x): return float(np.sum((x-target)**2))
    def g(x):
        pair=x[:-1]+0.55*x[1:]-0.35
        return np.r_[pair, np.sum(x*x)-0.27*d]
    return Problem(f'LargeChain{d}',d,[[-1.5,1.5]]*d,f,g,[2]*max(d-1,0)+[max(10,d)])

def scalable_shell(dim=100):
    d=int(dim); c=np.full(d,0.2)
    def f(x): return float(np.mean((x-c)**2)+0.01*np.mean(np.sin(6*x)**2))
    def g(x):
        r=np.mean((x-c)**2)
        return np.array([0.08-r, r-0.55, 0.03-np.mean(x)])
    return Problem(f'LargeShell{d}',d,[[-1.5,1.5]]*d,f,g,[1,1,2])

def engineering_problems():
    return [welded_beam(), pressure_vessel(), tension_spring(), cantilever_beam()]

def scalable_problems(dims=(100,500,1000)):
    out=[]
    for d in dims:
        out.extend([scalable_chain(d),scalable_shell(d)])
    return out
