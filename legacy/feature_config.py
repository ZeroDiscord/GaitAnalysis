import hashlib
import json
from typing import List, Optional

class FeatureConfig:
    """
    Configuration mapping class directly linking the mathematical extractions 
    implemented in EDA/eda_features.py + gait_phase.py 
    into the PyTorch dataloader tensors.

    The 20 features implemented here are based on the Random Forest
    Permutation Importance Plot to ensure maximum model performance.
    """
    def __init__(self,
                 legacy_mode: bool = False,
                 include_base_features: bool = True,
                 include_time_domain: bool = True,
                 include_freq_domain: bool = True,
                 include_gait_cycle: bool = True,
                 enable_pca: bool = False,
                 pca_components: int = 10,
                 enable_ica: bool = False,
                 ica_components: int = 8,
                 enable_feature_caching: bool = True,
                 cache_dir: str = 'cache/features/'):
        
        self.legacy_mode = legacy_mode
        self.include_base_features = include_base_features
        self.include_time_domain = include_time_domain
        self.include_freq_domain = include_freq_domain
        self.include_gait_cycle = include_gait_cycle
        
        # Base Features (4 from diagram: e_ago, stiffness, torque, plus e_ant, gait phase)
        self.base_features = ['e_ant', 'e_ago', 'torque', 'stiffness', 'gait_phase']
        
        # Time Domain Features based heavily on the feature importance chart
        self.time_domain_features = [
            'ssc_ant',
            'wl_ant',
            'zcr_ago',
            'zcr_ant',
            'iemg_ant',
            'variance_ant',
            'wl_ago',
            'mav_ant',
            'rms_ant',
            'variance_ago'
        ]
        
        # Frequency Domain Features based heavily on feature importance chart
        self.freq_domain_features = [
            'spectral_entropy_ago',
            'mean_freq_ago',
            'mean_freq_ant',
            'total_power_ago',
            'peak_freq_ant',
            'median_freq_ago'
        ]
        
        # Gait Cycle Features based heavily on feature importance chart
        self.gait_cycle_features = [
            'gp_propulsion_phase_duration'
        ]

        self.enable_pca = enable_pca
        self.pca_components = pca_components
        self.enable_ica = enable_ica
        self.ica_components = ica_components
        
        self.enable_feature_caching = enable_feature_caching
        self.cache_dir = cache_dir

    @classmethod
    def create_legacy(cls):
        """Creates a Legacy mode Feature Config (5 base features only)."""
        return cls(legacy_mode=True, 
                   include_base_features=True, 
                   include_time_domain=False, 
                   include_freq_domain=False, 
                   include_gait_cycle=False,
                   enable_pca=False,
                   enable_ica=False,
                   enable_feature_caching=False)

    def get_enabled_features(self) -> List[str]:
        """Returns the ordered list of feature names configured to be extracted."""
        if self.legacy_mode:
            return self.base_features

        features = []
        if self.include_base_features:
            features.extend(self.base_features)
            
        if self.include_time_domain:
            features.extend(self.time_domain_features)
            
        if self.include_freq_domain:
            features.extend(self.freq_domain_features)
            
        if self.include_gait_cycle:
            features.extend(self.gait_cycle_features)
            
        return features

    def get_total_feature_count(self) -> int:
        """Helper to inform PyTorch model initialization of input dimensionality."""
        return len(self.get_enabled_features())

    def get_config_hash(self) -> str:
        """Returns a deterministic SHA256 string representing this configuration for caching."""
        config_dict = {
            'legacy_mode': self.legacy_mode,
            'include_base_features': self.include_base_features,
            'include_time_domain': self.include_time_domain,
            'include_freq_domain': self.include_freq_domain,
            'include_gait_cycle': self.include_gait_cycle,
            'features': self.get_enabled_features(),
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[0:8]
