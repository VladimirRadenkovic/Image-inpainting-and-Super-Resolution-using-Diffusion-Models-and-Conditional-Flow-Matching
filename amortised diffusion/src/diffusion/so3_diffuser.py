"""SO(3) diffusion methods."""
import numpy as np
import os
import src.diffusion.utils as du
import logging
import torch


def igso3_expansion(omega, eps, L=1000):
    """Truncated sum of IGSO(3) distribution.

    This function approximates the power series in equation 5 of
    "DENOISING DIFFUSION PROBABILISTIC MODELS ON SO(3) FOR ROTATIONAL
    ALIGNMENT"
    Leach et al. 2022

    This expression diverges from the expression in Leach in that here, eps =
    sqrt(2) * eps_leach, if eps_leach were the scale parameter of the IGSO(3).

    With this reparameterization, IGSO(3) agrees with the Brownian motion on
    SO(3) with t=eps^2.

    Args:
        omega (float): Rotation of Euler vector, i.e., the angle of rotation.
        eps (float): Standard deviation of IGSO(3). Related to the time 
                     parameter of the associated Brownian motion on SO(3) by 
                     `eps^2 = t`.
        L (int, optional): Truncation level for the power series. Defaults to 1000.
    
    Returns:
        float: Truncated sum of the power series approximation to the pdf of the IGSO(3) distribution.
    """
    p = 0
    for l in range(L):
        p += (2*l + 1) * np.exp(-l*(l+1)*eps**2/2) * np.sin(omega*(l+1/2)) / np.sin(omega/2)
    return p


def density(expansion, omega, marginal=True):
    """
    Calculate the IGSO(3) density using a truncated approximation of the 
    power series expansion of the heat kernel on SO(3).

    This function can calculate either the marginal density over the angle 
    of rotation, or the normalized density on SO(3) for a rotation with 
    angle `omega`.

    Args:
        expansion (float): Truncated approximation of the power series in 
                           the IGSO(3) density.
        omega (float): Length of an Euler vector (i.e., angle of rotation).
        marginal (bool, optional): If True, calculates the marginal density 
                                    over the angle of rotation. If False, 
                                    includes normalization to calculate the 
                                    density on SO(3) for a rotation with 
                                    angle `omega`. Defaults to True.

    Returns:
        float: The calculated IGSO(3) density.
    """
    if marginal:
        # if marginal, density over [0, pi], else over SO(3)
        return expansion * (1-np.cos(omega))/np.pi
    else:
        # the constant factor doesn't affect any actual calculations though
        return expansion / 8 / np.pi**2


def score(exp, omega, eps, L=1000):  # score of density over SO(3)
    """score uses the quotient rule to compute the scaling factor for the score
    of the IGSO(3) density.

    This function is used within the Diffuser class to when computing the score
    as an element of the tangent space of SO(3).

    This uses the quotient rule of calculus, and take the derivative of the
    log:
        d hi(x)/lo(x) = (lo(x) d hi(x)/dx - hi(x) d lo(x)/dx) / lo(x)^2
    and
        d log expansion(x) / dx = (d expansion(x)/ dx) / expansion(x)

    Args:
        exp (float): Truncated expansion of the power series in the IGSO(3) density.
        omega (float): Length of an Euler vector (i.e., angle of rotation).
        eps (float): Scale parameter for IGSO(3). The scaling factor differs from that in Leach by a factor of sqrt(2).
        L (int, optional): Truncation level. Defaults to 1000.

    Returns:
        float: The scaling factor of the IGSO3(omega; eps)/(1-cos(omega)).
    """
    dSigma = 0
    for l in range(L):
        hi = np.sin(omega * (l + 1 / 2))
        dhi = (l + 1 / 2) * np.cos(omega * (l + 1 / 2))
        lo = np.sin(omega / 2)
        dlo = 1 / 2 * np.cos(omega / 2)
        dSigma += (2 * l + 1) * np.exp(-l * (l + 1) * eps**2/2) * (lo * dhi - hi * dlo) / lo ** 2
    return dSigma / exp


class SO3Diffuser:
    """A class for generating and diffusing rotations in IGSO(3).

        Args:
            so3_conf: Configuration object containing parameters for the SO3Diffuser.

        Attributes:
            schedule (str): Schedule for choosing sigma values.
            min_sigma (float): Minimum sigma value.
            max_sigma (float): Maximum sigma value.
            num_sigma (int): Number of sigma values.
            _log (Logger): Logger object for logging information.
            discrete_omega (numpy.ndarray): Discretized omegas for calculating CDFs.
            _pdf (numpy.ndarray): Precomputed pdf values.
            _cdf (numpy.ndarray): Precomputed cdf values.
            _score_norms (numpy.ndarray): Precomputed score norms.
            _score_scaling (numpy.ndarray): Scaling factors for score vectors.
    """
    def __init__(self, schedule, min_sigma, max_sigma, num_sigma, num_omega, cache_dir):
        self.schedule = schedule

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        self.num_sigma = num_sigma
        self._log = logging.getLogger(__name__)

        # Discretize omegas for calculating CDFs. Skip omega=0.
        self.discrete_omega = np.linspace(0, np.pi, num_omega+1)[1:]

        # Precompute IGSO3 values.
        replace_period = lambda x: str(x).replace('.', '_')
        cache_dir = os.path.join(
            cache_dir,
            f'eps_{num_sigma}_omega_{num_omega}_min_sigma_{replace_period(min_sigma)}_max_sigma_{replace_period(max_sigma)}_schedule_{schedule}'
        )

        # If cache directory doesn't exist, create it
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        pdf_cache = os.path.join(cache_dir, 'pdf_vals.npy')
        cdf_cache = os.path.join(cache_dir, 'cdf_vals.npy')
        score_norms_cache = os.path.join(cache_dir, 'score_norms.npy')

        if os.path.exists(pdf_cache) and os.path.exists(cdf_cache) and os.path.exists(score_norms_cache):
            self._log.info(f'Using cached IGSO3 in {cache_dir}')
            self._pdf = np.load(pdf_cache)
            self._cdf = np.load(cdf_cache)
            self._score_norms = np.load(score_norms_cache)
        else:
            self._log.info(f'Computing IGSO3. Saving in {cache_dir}')
            # compute the expansion of the power series
            exp_vals = np.asarray(
                [igso3_expansion(self.discrete_omega, sigma) for sigma in self.discrete_sigma])
            # Compute the pdf and cdf values for the marginal distribution of the angle
            # of rotation (which is needed for sampling)
            self._pdf  = np.asarray(
                [density(x, self.discrete_omega, marginal=True) for x in exp_vals])
            self._cdf = np.asarray(
                [pdf.cumsum() / num_omega * np.pi for pdf in self._pdf])

            # Compute the norms of the scores.  This are used to scale the rotation axis when
            # computing the score as a vector.
            self._score_norms = np.asarray(
                [score(exp_vals[i], self.discrete_omega, x) for i, x in enumerate(self.discrete_sigma)])

            # Cache the precomputed values
            np.save(pdf_cache, self._pdf)
            np.save(cdf_cache, self._cdf)
            np.save(score_norms_cache, self._score_norms)

        self._score_scaling = np.sqrt(np.abs(
            np.sum(
                self._score_norms**2 * self._pdf, axis=-1) / np.sum(
                    self._pdf, axis=-1)
        )) / np.sqrt(3)

    @property
    def discrete_sigma(self):
        return self.sigma(
            np.linspace(0.0, 1.0, self.num_sigma)
        )

    def sigma_idx(self, sigma: np.ndarray):
        """Calculates the index for discretized sigma during IGSO(3) initialization."""
        return np.digitize(sigma, self.discrete_sigma) - 1

    def sigma(self, t: np.ndarray):
        """Extract \sigma(t) corresponding to chosen sigma schedule."""
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        if self.schedule == 'logarithmic':
            return np.log(t * np.exp(self.max_sigma) + (1 - t) * np.exp(self.min_sigma))
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')

    def diffusion_coef(self, t):
        """Compute diffusion coefficient (g_t)."""
        if self.schedule == 'logarithmic':
            g_t = np.sqrt(
                2 * (np.exp(self.max_sigma) - np.exp(self.min_sigma)) * self.sigma(t) / np.exp(self.sigma(t))
            )
        #!TODO: Add other schedules (Hoogeboom et al.)
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')
        return g_t

    def t_to_idx(self, t: np.ndarray):
        """Helper function to go from time t to corresponding sigma_idx."""
        return self.sigma_idx(self.sigma(t))

    def sample_igso3(
            self,
            t: float,
            n_samples: float=1):
        """Uses the inverse cdf to sample an angle of rotation from IGSO(3).

        Args:
            t (float): Continuous time in the range [0, 1].
            n_samples (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            numpy.ndarray: An array of angles of rotation sampled from the IGSO(3) distribution.
                            The shape of the array is (n_samples,).
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x = np.random.rand(n_samples)
        return np.interp(x, self._cdf[self.t_to_idx(t)], self.discrete_omega)

    def sample(
            self,
            t: float,
            n_samples: float=1):
        """Generates rotation vector(s) from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            n_sample: number of samples to generate.

        Returns:
            [n_samples, 3] axis-angle rotation vectors sampled from IGSO(3).
        """
        x = np.random.randn(n_samples, 3)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return x * self.sample_igso3(t, n_samples=n_samples)[:, None]

    def sample_ref(self, n_samples: float=1):
        return self.sample(1, n_samples=n_samples)

    def score(
            self,
            vec: np.ndarray,
            t: float,
            eps: float=1e-6
        ):
        """Computes the score of IGSO(3) density as a rotation vector.

        Args:
            vec: [..., 3] array of axis-angle rotation vectors.
            t: continuous time in [0, 1].

        Returns:
            [..., 3] score vector in the direction of the sampled vector with
            magnitude given by _score_norms.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        omega = np.linalg.norm(vec, axis=-1)
        return np.interp(
            omega, self.discrete_omega, self._score_norms[self.t_to_idx(t)]
        )[:, None] * vec / (omega[:, None] + eps)

    def torch_score(
            self,
            vec: torch.tensor,
            t: torch.tensor,
            eps: float=1e-6
        ):
        """Computes the score of IGSO(3) density as a rotation vector.

        Same as score function but uses pytorch and performs a look-up.

        Args:
            vec: [..., 3] array of axis-angle rotation vectors.
            t: continuous time in [0, 1].

        Returns:
            [..., 3] score vector in the direction of the sampled vector with
            magnitude given by _score_norms.
        """
        omega = torch.linalg.norm(vec, dim=-1)
        score_norms_t = self._score_norms[self.t_to_idx(du.move_to_np(t))]
        score_norms_t = torch.tensor(score_norms_t).to(vec.device)
        omega_idx = torch.bucketize(
            omega, torch.tensor(self.discrete_omega[:-1]).to(vec.device))
        omega_score_t = torch.gather(
            score_norms_t, 1, omega_idx)
        return omega_score_t[..., None] * vec / (omega[..., None] + eps)

    def score_scaling(self, t: np.ndarray):
        """Calculates scaling used for scores during trianing."""
        return self._score_scaling[self.t_to_idx(t)]

    def forward_marginal(self, rot_0: np.ndarray, t: float):
        """Samples from the forward diffusion process at time index t.

        Args:
            rot_0: [..., 3] initial rotations.
            t: continuous time in [0, 1].

        Returns:
            rot_t: [..., 3] noised rotation vectors.
            rot_score: [..., 3] score of rot_t as a rotation vector.
        """
        n_samples = np.cumprod(rot_0.shape[:-1])[-1]
        sampled_rots = self.sample(t, n_samples=n_samples)
        rot_score = self.score(sampled_rots, t).reshape(rot_0.shape)

        # Right multiply.
        rot_t = du.compose_rotvec(rot_0, sampled_rots).reshape(rot_0.shape)
        return rot_t, rot_score

    def reverse(
            self,
            rot_t: np.ndarray,
            score_t: np.ndarray,
            t: float,
            dt: float,
            mask: np.ndarray=None,
            noise_scale: float=1.0,
            ):
        """Simulates the reverse SDE for 1 step using the Geodesic random walk.

        Args:
            rot_t: [..., 3] current rotations at time t.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            add_noise: set False to set diffusion coefficent to 0.
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] rotation vector at next step.
        """
        if not np.isscalar(t): raise ValueError(f'{t} must be a scalar.')

        g_t = self.diffusion_coef(t)
        z = noise_scale * np.random.normal(size=score_t.shape)
        perturb = (g_t ** 2) * score_t * dt + g_t * np.sqrt(dt) * z

        if mask is not None: perturb *= mask[..., None]
        n_samples = np.cumprod(rot_t.shape[:-1])[-1]

        # Right multiply.
        rot_t_1 = du.compose_rotvec(
            rot_t.reshape(n_samples, 3),
            perturb.reshape(n_samples, 3)
        ).reshape(rot_t.shape)
        return rot_t_1
    
