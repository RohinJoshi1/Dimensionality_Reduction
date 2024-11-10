import numpy as np
from typing import Union, Optional
import warnings

class RandomProjector:
    """
    Implements Random Projection for dimensionality reduction of embedding vectors.
    Uses either Gaussian or sparse random projection matrices.
    """
    
    def __init__(
        self, 
        target_dim: int,
        projection_type: str = 'gaussian',
        density: float = 1/3,
        random_state: Optional[int] = None
    ):
        """
        Initialize the random projector.
        
        Args:
            target_dim: The target dimensionality after projection
            projection_type: 'gaussian' or 'sparse' random projection
            density: Density of the sparse random projection matrix (if using sparse)
            random_state: Random seed for reproducibility
        """
        self.target_dim = target_dim
        self.projection_type = projection_type.lower()
        self.density = density
        self.random_state = random_state
        self.projection_matrix = None
        self.input_dim = None
        
        if self.projection_type not in ['gaussian', 'sparse']:
            raise ValueError("projection_type must be either 'gaussian' or 'sparse'")
            
        if not (0 < density <= 1):
            raise ValueError("density must be in (0, 1]")
    
    def _init_gaussian_matrix(self, input_dim: int) -> None:
        """Initialize Gaussian random projection matrix."""
        rng = np.random.RandomState(self.random_state)
        self.projection_matrix = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(self.target_dim),
            size=(input_dim, self.target_dim)
        )
    
    def _init_sparse_matrix(self, input_dim: int) -> None:
        """Initialize sparse random projection matrix using the method from Achlioptas (2003)."""
        rng = np.random.RandomState(self.random_state)
        
        # Initialize sparse matrix
        self.projection_matrix = np.zeros((input_dim, self.target_dim))
        
        # Calculate number of non-zero elements per column
        nnz = int(input_dim * self.density)
        
        # Generate sparse matrix
        for j in range(self.target_dim):
            # Choose random positions for non-zero elements
            indices = rng.choice(input_dim, size=nnz, replace=False)
            values = rng.choice([-1, 1], size=nnz)
            self.projection_matrix[indices, j] = values
            
        # Scale the matrix
        self.projection_matrix *= 1.0 / np.sqrt(nnz)
    
    def fit(self, X: Union[np.ndarray, list]) -> 'RandomProjector':
        """
        Fit the random projection matrix to the input dimensionality.
        
        Args:
            X: Input array or list of arrays/lists with shape (n_samples, n_features)
        
        Returns:
            self: Returns the instance itself
        """
        if isinstance(X, list):
            X = np.array(X)
            
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        self.input_dim = X.shape[1]
        
        if self.target_dim >= self.input_dim:
            warnings.warn(
                "Target dimensionality is greater than or equal to input dimensionality. "
                "No dimension reduction will occur."
            )
            
        if self.projection_type == 'gaussian':
            self._init_gaussian_matrix(self.input_dim)
        else:
            self._init_sparse_matrix(self.input_dim)
            
        return self
    
    def transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Apply dimension reduction to X.
        
        Args:
            X: Array or list of arrays/lists with shape (n_samples, n_features)
            
        Returns:
            X_new: Projected array with shape (n_samples, target_dim)
        """
        if self.projection_matrix is None:
            raise ValueError("Projector must be fitted before calling transform")
            
        if isinstance(X, list):
            X = np.array(X)
            
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimensionality {X.shape[1]} does not match "
                f"fitted dimensionality {self.input_dim}"
            )
            
        return np.dot(X, self.projection_matrix)
    
    def fit_transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Fit the model with X and apply the dimensionality reduction on X.
        
        Args:
            X: Input array or list of arrays/lists
            
        Returns:
            X_new: Projected array
        """
        return self.fit(X).transform(X)
