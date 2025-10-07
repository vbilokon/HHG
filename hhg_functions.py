import numpy as np
from quspin.operators import hamiltonian, exp_op

def diff2(t: np.ndarray, y: np.array) -> np.ndarray:
    """
    Compute the second derivative using finite differences.
    
    Uses central differences for interior points and forward/backward 
    differences for boundary points.
    
    Parameters
    ----------
    t : np.ndarray
        Time or spatial coordinate array (should be uniformly spaced)
    y : np.ndarray  
        Function values corresponding to t coordinate
        
    Returns
    -------
    np.ndarray
        Second derivative values with same shape as input y
    """

    # Input validation
    if len(t) != len(y):
        raise ValueError(f"Arrays must have same length: t={len(t)}, y={len(y)}")
    
    if len(t) < 3:
        raise ValueError("Need at least 3 points to compute second derivative")
        
    dt = t[1] - t[0]   # calculate time step
    res = np.empty_like(y, dtype=float)   # pre-allocate result array

    # first element
    res[0] = y[2] - 2*y[1] + y[0]   # forward difference
    # middle elements
    res[1:-1] = y[2:] - 2*y[1:-1] + y[:-2]   # central difference
    # last element
    res[-1] = y[-1] - 2*y[-2] + y[-3]   # backward difference

    return res / dt**2



def magnetic_pulse(t: float, Omega: float, N_cycles: int, B: float) -> float:
    """
    Generate a magnetic field pulse: B_x(t) = B sin²(ωt/2N_cyc) cos(ωt)
    for 0 < t < T_f = 2πN_cyc/ω

    Parameters
    ----------
    t : float 
        Time point at which to evaluate the pulse
    Omega : float
        Angular frequency 
    N_cycles : int
        Number of cycles in the pulse
    B : float
        Peak amplitude of the magnetic field

    Returns
    -------
    float or np.ndarray
        Magnetic field value at time t
    """

    # Input validation
    if Omega <= 0:
        raise ValueError(f"Angular frequency must be positive, got {Omega}")
    if N_cycles <= 0:
        raise ValueError(f"Number of cycles must be positive, got {N_cycles}")

    # calculate pulse duration
    T_f = 2 * np.pi * N_cycles / Omega
    if 0 <= t <= T_f:
        envelope = np.sin(Omega * t / (2 * N_cycles))**2
        carrier = np.cos(Omega * t)
        return B * envelope * carrier 
    else:
        return 0.0


def ising1d_hamiltonian(basis: any, L: int, J: float, h: float, B: float, Omega: float, N_cycles: int, bc: int = 1, disable_checks: bool = True) -> any:
    """
    Construct time-dependent Ising model Hamiltonian with magnetic field pulse.

    Parameters
    ----------
    basis : quspin.basis object
        Quantum basis for the spin system
    L : int
        Number of spins in the chain
    J : float
        Nearest-neighbor coupling strength 
    h : float
        Static magnetic field strength in z-direction
    B : float
        Peak amplitude B of the time-dependent magnetic field pulse
    Omega : float
        Angular frequency of the magnetic field pulse 
    N_cycles : int
        Number of cycles in the pulse 
    bc : int
        0 -- open boundary conditions, 1 -- periodic boundary conditions (default: periodic)
    disable_checks : bool
        Whether to disable QuSpin symmetry/hermiticity checks for performance (default: True)

    Returns
    -------
    quspin.operators.hamiltonian
        Time-dependent Hamiltonian object
    """

    # Input validation
    if L <= 0:
        raise ValueError(f"Chain length must be positive, got {L}")
    
    if Omega <= 0:
        raise ValueError(f"Pulse frequency must be positive, got {Omega}")
        
    if N_cycles <= 0:
        raise ValueError(f"Number of pulse cycles must be positive, got {N_cycles}")
    

    if bc not in {0, 1}:
      raise ValueError(f"Boundary condition must be 0 (open) or 1 (periodic), got {bc}")

    # construct nearest-neighbor coupling terms
    spin_coupling = [[-J, i, (i + 1) % L] for i in range(L if bc else L - 1)]
    
    # Define static and dynamic magnetic fields
    stat_field = [[-h, i] for i in range(L)]   # longitudinal, z-direction
    dynamic_field = [[-1, i] for i in range(L)]   # transverse, x-direction

    # pulse parameters for time-dependent Hamiltonian
    pulse_args = [Omega, N_cycles, B]
    
    # Hamiltonian static terms
    static = [
          ["zz", spin_coupling], # spin coupling
          ["z", stat_field] # static magnetic field
          ]
    # Hamiltonian dynamic terms
    dynamic = [["x", dynamic_field, magnetic_pulse, pulse_args]]

    if disable_checks:
        no_checks = {
            'check_pcon': False,    # skip particle conservation check
            'check_symm': False,    # skip symmetry check  
            'check_herm': False     # skip hermiticity check
        }
    
    # construct the Hamiltonian
    return hamiltonian(static, dynamic, basis=basis, dtype=np.float64, **no_checks)


def create_translational_symmetry(Lx: int, Ly: int):
    Nsites = Lx * Ly
    s = np.arange(Nsites)  # sites [0,1,2,....]
    x = s % Lx  # x positions for sites
    y = s // Lx  # y positions for sites
    T_x = (x + 1) % Lx + Lx * y  # translation along x-direction
    T_y = x + Lx * ((y + 1) % Ly)  # translation along y-direction
    return T_x, T_y


def ising2d_hamiltonian(
    basis: any,
    Lx: int, 
    Ly: int,
    Nsites: int, 
    J: float, 
    h: float, 
    B: float, 
    Omega: float, 
    N_cycles: int, 
    bc: int = 1, 
    disable_checks: bool = True
) -> any:
    
    """
    Construct time-dependent Ising model Hamiltonian with magnetic field pulse.

    Parameters
    ----------
    basis : quspin.basis object
        Quantum basis for the spin system
    Nsites : int
        Total number of spins in the system
    J : float
        Nearest-neighbor coupling strength 
    h : float
        Static magnetic field strength in z-direction
    B : float
        Peak amplitude B of the time-dependent magnetic field pulse
    Omega : float
        Angular frequency of the magnetic field pulse 
    N_cycles : int
        Number of cycles in the pulse 
    bc : int
        0 -- open boundary conditions, 1 -- periodic boundary conditions (default: periodic)
    disable_checks : bool
        Whether to disable QuSpin symmetry/hermiticity checks for performance (default: True)

    Returns
    -------
    quspin.operators.hamiltonian
        Time-dependent Hamiltonian object
    """

    # Input validation
    if Nsites <= 0:
        raise ValueError(f"Number of sites must be positive, got {Nsites}")
    
    if Omega <= 0:
        raise ValueError(f"Pulse frequency must be positive, got {Omega}")
        
    if N_cycles <= 0:
        raise ValueError(f"Number of pulse cycles must be positive, got {N_cycles}")
    

    if bc not in {0, 1}:
      raise ValueError(f"Boundary condition must be 0 (open) or 1 (periodic), got {bc}")

    T_x, T_y = create_translational_symmetry(Lx, Ly)

    # construct nearest-neighbor coupling terms
    spin_coupling = [[-J, i, T_x[i]] for i in range(Nsites)] + [
        [-1.0, i, T_y[i]] for i in range(Nsites)
    ]
    
    # Define static and dynamic magnetic fields
    stat_field = [[-h, i] for i in range(Nsites)]   # longitudinal, z-direction
    dynamic_field = [[-1, i] for i in range(Nsites)]   # transverse, x-direction

    # pulse parameters for time-dependent Hamiltonian
    pulse_args = [Omega, N_cycles, B]
    
    # Hamiltonian static terms
    static = [
          ["zz", spin_coupling], # spin coupling
          ["z", stat_field] # static magnetic field
          ]
    # Hamiltonian dynamic terms
    dynamic = [["x", dynamic_field, magnetic_pulse, pulse_args]]

    if disable_checks:
        no_checks = {
            'check_pcon': False,    # skip particle conservation check
            'check_symm': False,    # skip symmetry check  
            'check_herm': False     # skip hermiticity check
        }
    
    # construct the Hamiltonian
    return hamiltonian(static, dynamic, basis=basis, dtype=np.float64, **no_checks)


def magnetization(Nsites: int, basis: any, operator: str = "x") -> any:
    """
    Create magnetization operator M for specified operator type.
    
    Parameters
    ----------
    Nsites : int
        Total number of spins in the system
    basis : quspin.basis object
        Quantum basis for the spin system
    operator : str
        Pauli operator type: 'x', 'y', 'z'
        
    Returns
    -------
    quspin.operators.hamiltonian
        Magnetization operator as QuSpin Hamiltonian object
    """

    # Input validation
    if not isinstance(Nsites, int) or Nsites <= 0:
        raise ValueError(f"Nsites must be a positive integer, got {Nsites}")
    
    # Validate operator
    valid_operators = {'x', 'y', 'z'}
    if operator not in valid_operators:
        raise ValueError(
            f"Operator must be one of {valid_operators}, got '{operator}'"
        )

    # create operator list
    m = [[1, i] for i in range(Nsites)]

    # build static terms
    static = [
      [operator, m]
      ]

    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    
    return hamiltonian(static, [], basis=basis, dtype=np.float64, **no_checks)


def blackman_window_paper(t: float, Tf: float) -> float:
    """
    Calculate Blackman function values

    Parameters
    ----------
    t : float 
        Time point at which to evaluate the window (0 ≤ t ≤ Tf)
    Tf : float
        Total duration of the window 
        
    Returns
    -------
    float 
        Window function value    
    """
    return 0.42 - 0.5 * np.cos(2 * np.pi * t / Tf) + 0.08 * np.cos(4 * np.pi * t / Tf)


# use for time-independent hamiltonian
def sys_evolve_time_independent_ham(hamiltonian, psi0, t):

  # use exp_op to get the evolution operator
  U = exp_op(hamiltonian, a=-1j, start=t.min(), stop=t.max(), num=len(t), iterate=True)

  psi_t = U.dot(psi0) # get generator psi_t for time evolved state

  return psi_t


def blackman_window(N):
  n = np.arange(N)
  return 0.42 - 0.5 * np.cos(2*np.pi*n/(N-1)) + 0.08 * np.cos(4*np.pi*n/(N-1))